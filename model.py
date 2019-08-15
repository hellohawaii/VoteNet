import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_nms'))
import tensorflow as tf
from tensorpack import *
import numpy as np
from tensorpack.tfutils import get_current_tower_context, gradproc, optimizer, summary, varreplace
from utils import pointnet_sa_module, pointnet_fp_module, pts2box, pos_pts2box
from dataset import class_mean_size
from tf_nms3d import NMS3D
import config


class Model(ModelDesc):
    def inputs(self):
        return [
                tf.placeholder(tf.int32, [None,], 'data_idx'),
                tf.placeholder(tf.float32, [None, config.POINT_NUM , 3], 'points'),
                tf.placeholder(tf.float32, [None, None, 3], 'bboxes_xyz'),
                tf.placeholder(tf.float32, [None, None, 3], 'bboxes_lwh'),
                tf.placeholder(tf.int32, (None, None), 'semantic_labels_input'),
                tf.placeholder(tf.int32, (None, None), 'heading_labels_input'),
                tf.placeholder(tf.float32, (None, None), 'heading_residuals_input'),
                tf.placeholder(tf.int32, (None, None), 'size_labels_input'),
                tf.placeholder(tf.float32, (None, None, 3), 'size_residuals_input'),
                ]

    def build_graph(self, _, x, bboxes_xyz, bboxes_lwh, semantic_labels, heading_labels, heading_residuals, size_labels, size_residuals):
        l0_xyz = x
        l0_points = x

        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=64,
                                                           mlp=[64, 64, 128], mlp2=None, group_all=False, scope='sa1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=0.8, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa3')
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=1.2, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa4')
        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], scope='fp1')
        seeds_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], scope='fp2')
        seeds_xyz = l2_xyz

        # Voting Module layers
        offset = tf.reshape(tf.concat([seeds_xyz, seeds_points], 2), [-1, 256 + 3])
        units = [256, 256, 256 + 3]
        for i in range(len(units)):
            offset = FullyConnected('voting%d' % i, offset, units[i], activation=BNReLU if i < len(units) - 1 else None)
        offset = tf.reshape(offset, [-1, 1024, 256 + 3])

        # B * N * 3
        votes = tf.concat([seeds_xyz, seeds_points], 2) + offset
        votes_xyz = votes[:, :, :3]
        dist2center = tf.abs(tf.expand_dims(seeds_xyz, 2) - tf.expand_dims(bboxes_xyz, 1))
        surface_ind = tf.less(dist2center, tf.expand_dims(bboxes_lwh, 1) / 2.)  # B * N * BB * 3, bool
        surface_ind = tf.equal(tf.count_nonzero(surface_ind, -1), 3)  # B * N * BB
        surface_ind = tf.greater_equal(tf.count_nonzero(surface_ind, -1), 1)  # B * N, should be in at least one bbox

        dist2center_norm = tf.norm(dist2center, axis=-1)  # B * N * BB
        votes_assignment = tf.argmin(dist2center_norm, -1, output_type=tf.int32)  # B * N, int
        bboxes_xyz_votes_gt = tf.gather_nd(bboxes_xyz, tf.stack([
            tf.tile(tf.expand_dims(tf.range(tf.shape(votes_assignment)[0]), -1), [1, tf.shape(votes_assignment)[1]]),
            votes_assignment], 2))  # B * N * 3
        vote_reg_loss = tf.reduce_mean(tf.norm(votes_xyz - bboxes_xyz_votes_gt, ord=1, axis=-1) * tf.cast(surface_ind, tf.float32), name='vote_reg_loss')
        votes_points = votes[:, :, 3:]

        # Proposal Module layers
        # Farthest point sampling on seeds
        proposals_xyz, proposals_output, _ = pointnet_sa_module(votes_xyz, votes_points, npoint=config.PROPOSAL_NUM,
                                                                radius=0.3, nsample=64, mlp=[128, 128, 128],
                                                                mlp2=[128, 128, 5+2 * config.NH+4 * config.NS+config.NC],
                                                                group_all=False, scope='proposal',
                                                                sample_xyz=seeds_xyz)

        obj_cls_score = tf.identity(proposals_output[..., :2], 'obj_scores')

        nms_iou = tf.get_variable('nms_iou', shape=[], initializer=tf.constant_initializer(0.25), trainable=False)
        if not get_current_tower_context().is_training:

            def get_3d_bbox(box_size, heading_angle, center):
                batch_size = tf.shape(heading_angle)[0]
                c = tf.cos(heading_angle)
                s = tf.sin(heading_angle)
                zeros = tf.zeros_like(c)
                ones = tf.ones_like(c)
                rotation = tf.reshape(tf.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], -1), tf.stack([batch_size, -1, 3, 3]))
                l, w, h = box_size[..., 0], box_size[..., 1], box_size[..., 2]  # lwh(xzy) order!!!
                corners = tf.reshape(tf.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2,
                                               h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2,
                                               w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1),
                                     tf.stack([batch_size, -1, 3, 8]))
                return tf.einsum('ijkl,ijlm->ijmk', rotation, corners) + tf.expand_dims(center, 2)  # B * N * 8 * 3

            class_mean_size_tf = tf.constant(class_mean_size)
            size_cls_pred = tf.argmax(proposals_output[..., 5 + 2 * config.NH: 5 + 2 * config.NH + config.NS], axis=-1)
            size_cls_pred_onehot = tf.one_hot(size_cls_pred, depth=config.NS, axis=-1)  # B * N * NS
            size_residual_pred = tf.reduce_sum(tf.expand_dims(size_cls_pred_onehot, -1)
                                               * tf.reshape(proposals_output[..., 5+2 * config.NH + config.NS:5+2 * config.NH + 4 * config.NS], (-1, config.PROPOSAL_NUM, config.NS, 3)), axis=2)
            size_pred = tf.gather_nd(class_mean_size_tf, tf.expand_dims(size_cls_pred, -1)) * tf.maximum(1 + size_residual_pred, 1e-6)  # B * N * 3: size
            # with tf.control_dependencies([tf.print(size_pred[0, 0, 2])]):
            center_pred = proposals_xyz + proposals_output[..., 2:5]  # B * N * 3
            heading_cls_pred = tf.argmax(proposals_output[..., 5:5+config.NH], axis=-1)
            heading_cls_pred_onehot = tf.one_hot(heading_cls_pred, depth=config.NH, axis=-1)
            heading_residual_pred = tf.reduce_sum(heading_cls_pred_onehot
                                                  * proposals_output[..., 5 + config.NH:5+2 * config.NH], axis=2)
            heading_pred = tf.floormod((tf.cast(heading_cls_pred, tf.float32) * 2 + heading_residual_pred) * np.pi / config.NH, 2 * np.pi)

            # with tf.control_dependencies([tf.print(size_residual_pred[0, :10, :]), tf.print(size_pred[0, :10, :])]):
            bboxes = get_3d_bbox(size_pred, heading_pred, center_pred)  # B * N * 8 * 3,  lhw(xyz) order!!!

            # bbox_corners = tf.concat([bboxes[:, :, 6, :], bboxes[:, :, 0, :]], axis=-1)  # B * N * 6,  lhw(xyz) order!!!
            # with tf.control_dependencies([tf.print(bboxes[0, 0])]):
            nms_idx = NMS3D(bboxes, tf.reduce_max(proposals_output[..., -config.NC:], axis=-1), proposals_output[..., :2], nms_iou)  # Nnms * 2

            bboxes_pred = tf.gather_nd(bboxes, nms_idx, name='bboxes_pred')  # Nnms * 8 * 3
            class_scores_pred = tf.gather_nd(proposals_output[..., -config.NC:], nms_idx, name='class_scores_pred')  # Nnms * C
            batch_idx = tf.identity(nms_idx[:, 0], name='batch_idx')  # Nnms, this is used to identify between batches

            return

        # calculate positive and negative proposal idxes
        bboxes_xyz_gt = bboxes_xyz  # B * BB * 3
        bboxes_labels_gt = semantic_labels  # B * BB
        bboxes_heading_labels_gt = heading_labels
        bboxes_heading_residuals_gt = heading_residuals
        bboxes_size_labels_gt = size_labels
        bboxes_size_residuals_gt = size_residuals
        dist_mat = tf.norm(tf.expand_dims(proposals_xyz, 2) - tf.expand_dims(bboxes_xyz_gt, 1), axis=-1)  # B * PR * BB
        bboxes_assignment = tf.argmin(dist_mat, axis=-1)  # B * PR
        min_dist = tf.reduce_min(dist_mat, axis=-1)

        positive_idxes = tf.where(min_dist < config.POSITIVE_THRES)  # Np * 2
        # with tf.control_dependencies([tf.print(tf.shape(positive_idxes))]):
        negative_idxes = tf.where(min_dist > config.NEGATIVE_THRES)  # Nn * 2
        positive_gt_idxes = tf.stack([positive_idxes[:, 0], tf.gather_nd(bboxes_assignment, positive_idxes)], axis=1)

        # objectiveness loss
        pos_obj_cls_score = tf.gather_nd(obj_cls_score, positive_idxes)
        pos_obj_cls_gt = tf.ones([tf.shape(positive_idxes)[0]], dtype=tf.int32)
        neg_obj_cls_score = tf.gather_nd(obj_cls_score, negative_idxes)
        neg_obj_cls_gt = tf.zeros([tf.shape(negative_idxes)[0]], dtype=tf.int32)
        obj_cls_loss = tf.identity(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pos_obj_cls_score, labels=pos_obj_cls_gt))
                                   + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=neg_obj_cls_score, labels=neg_obj_cls_gt)), name='obj_cls_loss')
        obj_correct = tf.concat([tf.cast(tf.nn.in_top_k(pos_obj_cls_score, pos_obj_cls_gt, 1), tf.float32),
                                 tf.cast(tf.nn.in_top_k(neg_obj_cls_score, neg_obj_cls_gt, 1), tf.float32)], axis=0, name='obj_correct')
        obj_accuracy = tf.reduce_mean(obj_correct, name='obj_accuracy')

        # center regression losses
        center_gt = tf.gather_nd(bboxes_xyz_gt, positive_gt_idxes)
        delta_predicted = tf.gather_nd(proposals_output[..., 2:5], positive_idxes)
        delta_gt = center_gt - tf.gather_nd(proposals_xyz, positive_idxes)
        center_loss = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=delta_gt, predictions=delta_predicted, reduction=tf.losses.Reduction.NONE), axis=-1))

        # Appendix A1: chamfer loss, assignment at least one bbox to each gt bbox
        bboxes_assignment_dual = tf.argmin(dist_mat, axis=1)  # B * BB
        batch_idx = tf.tile(tf.expand_dims(tf.range(tf.shape(bboxes_assignment_dual, out_type=tf.int64)[0]), axis=-1), [1, tf.shape(bboxes_assignment_dual)[1]])  # B * BB
        delta_gt_dual = bboxes_xyz_gt - tf.gather_nd(proposals_xyz, tf.stack([batch_idx, bboxes_assignment_dual], axis=-1))  # B * BB * 3
        delta_predicted_dual = tf.gather_nd(proposals_output[..., 2:5], tf.stack([batch_idx, bboxes_assignment_dual], axis=-1))  # B * BB * 3
        center_loss_dual = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=delta_gt_dual, predictions=delta_predicted_dual, reduction=tf.losses.Reduction.NONE), axis=-1))

        # add up
        center_loss += center_loss_dual

        # Heading loss
        heading_cls_gt = tf.gather_nd(bboxes_heading_labels_gt, positive_gt_idxes)
        heading_cls_score = tf.gather_nd(proposals_output[..., 5:5+config.NH], positive_idxes)
        heading_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=heading_cls_score, labels=heading_cls_gt))

        heading_cls_gt_onehot = tf.one_hot(heading_cls_gt,  depth=config.NH, on_value=1, off_value=0, axis=-1)  # Np * NH
        heading_residual_gt = tf.gather_nd(bboxes_heading_residuals_gt, positive_gt_idxes)  # Np
        heading_residual_predicted = tf.gather_nd(proposals_output[..., 5 + config.NH:5+2 * config.NH], positive_idxes)  # Np * NH
        heading_residual_loss = tf.losses.huber_loss(labels=heading_residual_gt,
                                                     predictions=tf.reduce_sum(heading_residual_predicted * tf.to_float(heading_cls_gt_onehot), axis=1), reduction=tf.losses.Reduction.MEAN)

        # Size loss
        size_cls_gt = tf.gather_nd(bboxes_size_labels_gt, positive_gt_idxes)
        size_cls_score = tf.gather_nd(proposals_output[..., 5+2 * config.NH:5+2 * config.NH + config.NS], positive_idxes)
        size_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=size_cls_score, labels=size_cls_gt))

        size_cls_gt_onehot = tf.one_hot(size_cls_gt, depth=config.NS, on_value=1, off_value=0, axis=-1)  # Np * NS
        size_cls_gt_onehot = tf.tile(tf.expand_dims(tf.to_float(size_cls_gt_onehot), -1), [1, 1, 3])  # Np * NS * 3
        size_residual_gt = tf.gather_nd(bboxes_size_residuals_gt, positive_gt_idxes)  # Np * 3
        size_residual_predicted = tf.reshape(tf.gather_nd(proposals_output[..., 5+2 * config.NH + config.NS:5+2 * config.NH + 4 * config.NS], positive_idxes), (-1, config.NS, 3))  # Np * NS * 3
        size_residual_loss = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=size_residual_gt,
                                                                               predictions=tf.reduce_sum(size_residual_predicted * tf.to_float(size_cls_gt_onehot), axis=1), reduction=tf.losses.Reduction.NONE), axis=-1))

        box_loss = center_loss + 0.1 * heading_cls_loss + heading_residual_loss + 0.1 * size_cls_loss + size_residual_loss

        # semantic loss
        sem_cls_score = tf.gather_nd(proposals_output[..., -config.NC:], positive_idxes)
        sem_cls_gt = tf.gather_nd(bboxes_labels_gt, positive_gt_idxes)  # Np
        sem_cls_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sem_cls_score, labels=sem_cls_gt),
            name='sem_cls_loss')
        sem_correct = tf.cast(tf.nn.in_top_k(sem_cls_score, sem_cls_gt, 1), tf.float32, name='sem_correct')
        sem_accuracy = tf.reduce_mean(sem_correct, name='sem_accuracy')

        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        summary.add_moving_summary(obj_accuracy, sem_accuracy)

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        # no weight decay
        # wd_cost = tf.multiply(1e-5,
        #                       regularize_cost('.*/W', tf.nn.l2_loss),
        #                       name='regularize_loss')
        total_cost = vote_reg_loss + 0.5 * obj_cls_loss + 1. * box_loss + 0.1 * sem_cls_loss
        total_cost = tf.identity(total_cost, name='total_cost')
        summary.add_moving_summary(total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        # This will also put the summary in tensorboard, stat.json and print in terminal,
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)

        return optimizer.apply_grad_processors(
            opt, [gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])


class PrimitiveModel(ModelDesc):
    def inputs(self):
        return [
                tf.placeholder(tf.int32, [None,], 'data_idx'),
                tf.placeholder(tf.float32, [None, config.POINT_NUM , 3], 'points'),
                ]

    def build_graph(self, _, x):
        l0_xyz = x
        l0_points = x

        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=2048, radius=0.2, nsample=64,
                                                           mlp=[64, 64, 128], mlp2=None, group_all=False, scope='sa1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=1024, radius=0.4, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=512, radius=0.8, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa3')
        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=256, radius=1.2, nsample=64,
                                                           mlp=[128, 128, 256], mlp2=None, group_all=False, scope='sa4')
        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], scope='fp1')
        seeds_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], scope='fp2')
        seeds_xyz = l2_xyz

        # Voting Module layers
        offset = tf.reshape(tf.concat([seeds_xyz, seeds_points], 2), [-1, 256 + 3])
        units = [256, 256, 256 + 3]
        for i in range(len(units)):
            offset = FullyConnected('voting%d' % i, offset, units[i], activation=BNReLU if i < len(units) - 1 else None)
        offset = tf.reshape(offset, [-1, 1024, 256 + 3])

        # B * N * 3
        votes = tf.concat([seeds_xyz, seeds_points], 2) + offset
        votes_xyz = votes[:, :, :3]
        '''
        dist2center = tf.abs(tf.expand_dims(seeds_xyz, 2) - tf.expand_dims(bboxes_xyz, 1))
        surface_ind = tf.less(dist2center, tf.expand_dims(bboxes_lwh, 1) / 2.)  # B * N * BB * 3, bool
        surface_ind = tf.equal(tf.count_nonzero(surface_ind, -1), 3)  # B * N * BB
        surface_ind = tf.greater_equal(tf.count_nonzero(surface_ind, -1), 1)  # B * N, should be in at least one bbox
        '''

        '''
        dist2center_norm = tf.norm(dist2center, axis=-1)  # B * N * BB
        votes_assignment = tf.argmin(dist2center_norm, -1, output_type=tf.int32)  # B * N, int
        bboxes_xyz_votes_gt = tf.gather_nd(bboxes_xyz, tf.stack([
            tf.tile(tf.expand_dims(tf.range(tf.shape(votes_assignment)[0]), -1), [1, tf.shape(votes_assignment)[1]]),
            votes_assignment], 2))  # B * N * 3
        vote_reg_loss = tf.reduce_mean(tf.norm(votes_xyz - bboxes_xyz_votes_gt, ord=1, axis=-1) * tf.cast(surface_ind, tf.float32), name='vote_reg_loss')
        '''
        votes_points = votes[:, :, 3:]

        # Proposal Module layers
        # Farthest point sampling on seeds
        proposals_xyz, proposals_output, _ = pointnet_sa_module(votes_xyz, votes_points, npoint=config.PROPOSAL_NUM,
                                                                radius=0.3, nsample=64, mlp=[128, 128, 128],
                                                                # mlp2=[128, 128, 5+2 * config.NH+4 * config.NS+config.NC],
                                                                mlp2=[128, 128, config.PARA_MUN],
                                                                group_all=False, scope='proposal',
                                                                sample_xyz=seeds_xyz)

        '''
        nms_iou = tf.get_variable('nms_iou', shape=[], initializer=tf.constant_initializer(0.25), trainable=False)
        '''
        if not get_current_tower_context().is_training:

            def get_3d_bbox(box_size, heading_angle, center):
                batch_size = tf.shape(heading_angle)[0]
                c = tf.cos(heading_angle)
                s = tf.sin(heading_angle)
                zeros = tf.zeros_like(c)
                ones = tf.ones_like(c)
                rotation = tf.reshape(tf.stack([c, zeros, s, zeros, ones, zeros, -s, zeros, c], -1), tf.stack([batch_size, -1, 3, 3]))
                l, w, h = box_size[..., 0], box_size[..., 1], box_size[..., 2]  # lwh(xzy) order!!!
                corners = tf.reshape(tf.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2,
                                               h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2,
                                               w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1),
                                     tf.stack([batch_size, -1, 3, 8]))
                return tf.einsum('ijkl,ijlm->ijmk', rotation, corners) + tf.expand_dims(center, 2)  # B * N * 8 * 3

            class_mean_size_tf = tf.constant(class_mean_size)
            size_cls_pred = tf.argmax(proposals_output[..., 5 + 2 * config.NH: 5 + 2 * config.NH + config.NS], axis=-1)
            size_cls_pred_onehot = tf.one_hot(size_cls_pred, depth=config.NS, axis=-1)  # B * N * NS
            size_residual_pred = tf.reduce_sum(tf.expand_dims(size_cls_pred_onehot, -1)
                                               * tf.reshape(proposals_output[..., 5+2 * config.NH + config.NS:5+2 * config.NH + 4 * config.NS], (-1, config.PROPOSAL_NUM, config.NS, 3)), axis=2)
            size_pred = tf.gather_nd(class_mean_size_tf, tf.expand_dims(size_cls_pred, -1)) * tf.maximum(1 + size_residual_pred, 1e-6)  # B * N * 3: size
            # with tf.control_dependencies([tf.print(size_pred[0, 0, 2])]):
            center_pred = proposals_xyz + proposals_output[..., 2:5]  # B * N * 3
            heading_cls_pred = tf.argmax(proposals_output[..., 5:5+config.NH], axis=-1)
            heading_cls_pred_onehot = tf.one_hot(heading_cls_pred, depth=config.NH, axis=-1)
            heading_residual_pred = tf.reduce_sum(heading_cls_pred_onehot
                                                  * proposals_output[..., 5 + config.NH:5+2 * config.NH], axis=2)
            heading_pred = tf.floormod((tf.cast(heading_cls_pred, tf.float32) * 2 + heading_residual_pred) * np.pi / config.NH, 2 * np.pi)

            # with tf.control_dependencies([tf.print(size_residual_pred[0, :10, :]), tf.print(size_pred[0, :10, :])]):
            bboxes = get_3d_bbox(size_pred, heading_pred, center_pred)  # B * N * 8 * 3,  lhw(xyz) order!!!

            # bbox_corners = tf.concat([bboxes[:, :, 6, :], bboxes[:, :, 0, :]], axis=-1)  # B * N * 6,  lhw(xyz) order!!!
            # with tf.control_dependencies([tf.print(bboxes[0, 0])]):
            nms_idx = NMS3D(bboxes, tf.reduce_max(proposals_output[..., -config.NC:], axis=-1), proposals_output[..., :2], nms_iou)  # Nnms * 2

            bboxes_pred = tf.gather_nd(bboxes, nms_idx, name='bboxes_pred')  # Nnms * 8 * 3
            class_scores_pred = tf.gather_nd(proposals_output[..., -config.NC:], nms_idx, name='class_scores_pred')  # Nnms * C
            batch_idx = tf.identity(nms_idx[:, 0], name='batch_idx')  # Nnms, this is used to identify between batches

            return

        # calculate positive and negative proposal idxes
        bboxes_xyz_gt = bboxes_xyz  # B * BB * 3
        '''
        bboxes_labels_gt = semantic_labels  # B * BB
        bboxes_heading_labels_gt = heading_labels
        bboxes_heading_residuals_gt = heading_residuals
        bboxes_size_labels_gt = size_labels
        bboxes_size_residuals_gt = size_residuals
        dist_mat = tf.norm(tf.expand_dims(proposals_xyz, 2) - tf.expand_dims(bboxes_xyz_gt, 1), axis=-1)  # B * PR * BB
        bboxes_assignment = tf.argmin(dist_mat, axis=-1)  # B * PR
        min_dist = tf.reduce_min(dist_mat, axis=-1)
        '''
        '''
        positive_idxes = tf.where(min_dist < config.POSITIVE_THRES)  # Np * 2
        # with tf.control_dependencies([tf.print(tf.shape(positive_idxes))]):
        negative_idxes = tf.where(min_dist > config.NEGATIVE_THRES)  # Nn * 2
        positive_gt_idxes = tf.stack([positive_idxes[:, 0], tf.gather_nd(bboxes_assignment, positive_idxes)], axis=1)

        # objectiveness loss
        pos_obj_cls_score = tf.gather_nd(obj_cls_score, positive_idxes)
        pos_obj_cls_gt = tf.ones([tf.shape(positive_idxes)[0]], dtype=tf.int32)
        neg_obj_cls_score = tf.gather_nd(obj_cls_score, negative_idxes)
        neg_obj_cls_gt = tf.zeros([tf.shape(negative_idxes)[0]], dtype=tf.int32)
        obj_cls_loss = tf.identity(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pos_obj_cls_score, labels=pos_obj_cls_gt))
                                   + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=neg_obj_cls_score, labels=neg_obj_cls_gt)), name='obj_cls_loss')
        obj_correct = tf.concat([tf.cast(tf.nn.in_top_k(pos_obj_cls_score, pos_obj_cls_gt, 1), tf.float32),
                                 tf.cast(tf.nn.in_top_k(neg_obj_cls_score, neg_obj_cls_gt, 1), tf.float32)], axis=0, name='obj_correct')
        obj_accuracy = tf.reduce_mean(obj_correct, name='obj_accuracy')
        '''
        '''
        # center regression losses
        center_gt = tf.gather_nd(bboxes_xyz_gt, positive_gt_idxes)
        delta_predicted = tf.gather_nd(proposals_output[..., 2:5], positive_idxes)
        delta_gt = center_gt - tf.gather_nd(proposals_xyz, positive_idxes)
        center_loss = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=delta_gt, predictions=delta_predicted, reduction=tf.losses.Reduction.NONE), axis=-1))
        '''
        '''
        # Appendix A1: chamfer loss, assignment at least one bbox to each gt bbox
        bboxes_assignment_dual = tf.argmin(dist_mat, axis=1)  # B * BB
        batch_idx = tf.tile(tf.expand_dims(tf.range(tf.shape(bboxes_assignment_dual, out_type=tf.int64)[0]), axis=-1), [1, tf.shape(bboxes_assignment_dual)[1]])  # B * BB
        delta_gt_dual = bboxes_xyz_gt - tf.gather_nd(proposals_xyz, tf.stack([batch_idx, bboxes_assignment_dual], axis=-1))  # B * BB * 3
        delta_predicted_dual = tf.gather_nd(proposals_output[..., 2:5], tf.stack([batch_idx, bboxes_assignment_dual], axis=-1))  # B * BB * 3
        center_loss_dual = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=delta_gt_dual, predictions=delta_predicted_dual, reduction=tf.losses.Reduction.NONE), axis=-1))

        # add up
        center_loss += center_loss_dual
        '''

        '''
        # Heading loss
        heading_cls_gt = tf.gather_nd(bboxes_heading_labels_gt, positive_gt_idxes)
        heading_cls_score = tf.gather_nd(proposals_output[..., 5:5+config.NH], positive_idxes)
        heading_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=heading_cls_score, labels=heading_cls_gt))

        heading_cls_gt_onehot = tf.one_hot(heading_cls_gt,  depth=config.NH, on_value=1, off_value=0, axis=-1)  # Np * NH
        heading_residual_gt = tf.gather_nd(bboxes_heading_residuals_gt, positive_gt_idxes)  # Np
        heading_residual_predicted = tf.gather_nd(proposals_output[..., 5 + config.NH:5+2 * config.NH], positive_idxes)  # Np * NH
        heading_residual_loss = tf.losses.huber_loss(labels=heading_residual_gt,
                                                     predictions=tf.reduce_sum(heading_residual_predicted * tf.to_float(heading_cls_gt_onehot), axis=1), reduction=tf.losses.Reduction.MEAN)

        # Size loss
        size_cls_gt = tf.gather_nd(bboxes_size_labels_gt, positive_gt_idxes)
        size_cls_score = tf.gather_nd(proposals_output[..., 5+2 * config.NH:5+2 * config.NH + config.NS], positive_idxes)
        size_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=size_cls_score, labels=size_cls_gt))

        size_cls_gt_onehot = tf.one_hot(size_cls_gt, depth=config.NS, on_value=1, off_value=0, axis=-1)  # Np * NS
        size_cls_gt_onehot = tf.tile(tf.expand_dims(tf.to_float(size_cls_gt_onehot), -1), [1, 1, 3])  # Np * NS * 3
        size_residual_gt = tf.gather_nd(bboxes_size_residuals_gt, positive_gt_idxes)  # Np * 3
        size_residual_predicted = tf.reshape(tf.gather_nd(proposals_output[..., 5+2 * config.NH + config.NS:5+2 * config.NH + 4 * config.NS], positive_idxes), (-1, config.NS, 3))  # Np * NS * 3
        size_residual_loss = tf.reduce_mean(tf.reduce_sum(tf.losses.huber_loss(labels=size_residual_gt,
                                                                               predictions=tf.reduce_sum(size_residual_predicted * tf.to_float(size_cls_gt_onehot), axis=1), reduction=tf.losses.Reduction.NONE), axis=-1))

        box_loss = center_loss + 0.1 * heading_cls_loss + heading_residual_loss + 0.1 * size_cls_loss + size_residual_loss

        # semantic loss
        sem_cls_score = tf.gather_nd(proposals_output[..., -config.NC:], positive_idxes)
        sem_cls_gt = tf.gather_nd(bboxes_labels_gt, positive_gt_idxes)  # Np
        sem_cls_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sem_cls_score, labels=sem_cls_gt),
            name='sem_cls_loss')
        sem_correct = tf.cast(tf.nn.in_top_k(sem_cls_score, sem_cls_gt, 1), tf.float32, name='sem_correct')
        sem_accuracy = tf.reduce_mean(sem_correct, name='sem_accuracy')
        '''

        '''
        # This will monitor training error & accuracy (in a moving average fashion). The value will be automatically
        # 1. written to tensosrboard
        # 2. written to stat.json
        # 3. printed after each epoch
        summary.add_moving_summary(obj_accuracy, sem_accuracy)
        '''

        # Use a regex to find parameters to apply weight decay.
        # Here we apply a weight decay on all W (weight matrix) of all fc layers
        # If you don't like regex, you can certainly define the cost in any other methods.
        # no weight decay
        # wd_cost = tf.multiply(1e-5,
        #                       regularize_cost('.*/W', tf.nn.l2_loss),
        #                       name='regularize_loss')

        ''''
        # bboxes_xyz(the gt of bounding box center): B * BB * 3 (BB is the num of bounding box)
        # votes_xys: B * N * 3 (N is the number of votes)
        # when compare bboxes_xyz and votes_xyz, expand dims to B * N * BB * 3
        # after expand_dims, become B * 1 * BB * 3, B * N * 1 * 3, Tensorflow will use broadcast
        # proposals_xyz: B * PR * 3 (PR is the num of proposal)
        '''
        # vote_reg_loss
        # refer to line 61 in model.py when writing these codes
        # TODO: Here, we use the nearest center as the GT, need to implement the version that using the closest box's
        #  center as GT
        vote2proposal_center = tf.abs(tf.expand_dims(votes_xyz, 2) - tf.expand_dims(proposals_xyz, 1))  # B * N * PR * 3
        vote2proposal_center_norm = tf.norm(vote2proposal_center, axis=-1)  # B * N * PR
        votes_assignment = tf.argmin(vote2proposal_center_norm, -1, output_type=tf.int32)  # B * N, int
        votes_gt = tf.gather_nd(proposals_xyz, tf.stack([
            tf.tile(input=tf.expand_dims(tf.range(tf.shape(votes_assignment)[0]), -1),
                    multiples=[1, tf.shape(votes_assignment)[1]]),
            votes_assignment
        ], 2))  # gather a B * N * 3 tensor from B * PR * 3 according to a B * N(votes_assignment)
        # the indices will be B * N * 2, indices[b, n] = [b, votes_assignment[b, n]]
        votes_gt_no_gradient = tf.stop_gradient(votes_gt)
        vote_reg_loss = tf.reduce_mean(tf.norm(votes_xyz - votes_gt_no_gradient, ord=1, axis=-1), name='vote_reg_loss')

        # obj_cls_loss & box_loss
        # First decide which box it is fit with for every point
        '''
        we assume that the proposals_output is B * PR * 11（2 objectness, 3 xyz, 3 lwh, 3 angles）
        data_idx is B * P * 3 (P is the number of total points)
        we want to get pts_assignment of B * P, pts_fit_loss of B * P
        '''
        # the rotation angle of each points relative to the proposal boxes
        alphas_star = -proposals_output[:, :, 8]  # B * PR
        betas_star = -proposals_output[:, :, 9]  # B * PR
        gammas_star = -proposals_output[:, :, 10]  # B * PR
        # referring to https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
        # rotation matrix
        # TODO: When do visualization, the meaning of the angles should be consistent
        b_pr = alphas_star.shape
        pr = alphas_star.shape[1]
        p = x.shape[1]
        r_alphas = tf.stack([tf.ones(b_pr), tf.zeros(b_pr), tf.zeros(b_pr),
                             tf.zeros(b_pr), tf.cos(alphas_star), -tf.sin(alphas_star),
                             tf.zeros(b_pr), tf.sin(alphas_star), tf.cos(alphas_star)], axis=2)
        r_betas = tf.stack([tf.cos(betas_star), tf.zeros(b_pr), tf.sin(betas_star),
                            tf.zeros(b_pr), tf.ones(b_pr), tf.zeros(b_pr),
                            -tf.sin(betas_star), tf.zeros(b_pr), tf.cos(betas_star)], axis=2)
        r_gammas = tf.stack([tf.cos(gammas_star), -tf.sin(gammas_star), tf.zeros(b_pr),
                             tf.sin(gammas_star), tf.cos(gammas_star), tf.zeros(b_pr),
                             tf.zeros(b_pr), tf.zeros(b_pr), tf.ones(b_pr)], axis=2)
        r_alphas = tf.reshape(r_alphas, shape=[b_pr[0], b_pr[1], 3, 3])
        r_betas = tf.reshape(r_betas, shape=[b_pr[0], b_pr[1], 3, 3])
        r_gammas = tf.reshape(r_gammas, shape=[b_pr[0], b_pr[1], 3, 3])
        r_matrix = tf.linalg.matmul(r_alphas, tf.linalg.matmul(r_betas, r_gammas))  # B * PR * 3 * 3
        r_matrix_expand = tf.expand_dims(r_matrix, axis=1)  # B * 1 * PR * 3 * 3
        r_matrix_tile = tf.tile(r_matrix_expand, multiples=[1, p, 1, 1, 1])  # B * P * PR * 3 * 3
        x_expand = tf.expand_dims(tf.expand_dims(x, axis=2), axis=-1)  # B * P * 1 * 3 * 1 from B * P * 3
        # here, we need column vector to do the multiplication,
        x_tile = tf.tile(x_expand, multiples=[1, 1, pr, 1, 1])  # B * P * PR * 3 * 1
        rotated_data_idx = tf.squeeze(tf.linalg.matmul(r_matrix_tile, x_tile))  # B * P * PR * 3
        # squeeze the additional axis to get the position tensor
        pts_to_box_assignment, pts_to_box_distance = pts2box(rotated_data_idx, proposals_output[:, :, 2:8])
        # both are B * P & B * P
        # obj_cls_loss
        # abandon the point at the origin
        origin_index = tf.equal(tf.count_nonzero(x, axis=-1), 3)  # B * P, origin point will be 1
        is_not_origin = tf.tile(tf.expand_dims(1-tf.cast(origin_index, dtype=tf.float32), axis=-1),
                                multiples=[1, 1, pr])  # B * P * PR
        proposal_fit_count = tf.count_nonzero(tf.math.multiply(tf.one_hot(pts_to_box_assignment, depth=pr),
                                                               is_not_origin), axis=1)  # B * PR
        obj_gt = tf.math.greater(proposal_fit_count, config.POSITIVE_THRES_NUM)  # B * PR, 1 or positive
        obj_cls_score_gt = tf.one_hot(obj_gt, depth=2, axis=-1)  # B * PR * 2
        obj_cls_score = tf.identity(proposals_output[..., :2], 'obj_scores')  # B * PR * 2
        obj_cls_loss = tf.identity(
            tf.reduce_min(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=obj_cls_score, labels=obj_cls_score_gt))
            , name='obj_cls_loss')

        # box_loss
        pos_pts_to_box_distance = pos_pts2box(rotated_data_idx, proposals_output[:, :, 2:8], obj_gt)
        box_loss = tf.math.reduce_sum(tf.math.multiply(pos_pts_to_box_distance, 1-origin_index))
        # total_cost = vote_reg_loss + 0.5 * obj_cls_loss + 1. * box_loss + 0.1 * sem_cls_loss
        total_cost = vote_reg_loss + 0.5 * obj_cls_loss + 1. * box_loss
        total_cost = tf.identity(total_cost, name='total_cost')
        summary.add_moving_summary(total_cost)

        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/W', ['histogram', 'rms']))
        # the function should return the total cost to be optimized
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        # This will also put the summary in tensorboard, stat.json and print in terminal,
        # but this time without moving average
        tf.summary.scalar('lr', lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.AdamOptimizer(lr)

        return optimizer.apply_grad_processors(
            opt, [gradproc.MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.5)),
                  gradproc.SummaryGradient()])

if __name__=='__main__':
   pass