import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import joblib
import pandas as pd
from utils.fix_joints_model import JointModel
import glob
from utils.consts import NODES, NODES2IND
from utils.helper_functions import fix_column_names, apply_savgol_filter
from tqdm import tqdm


# Define transformation functions for the model
def create_affine_transform_matrix(point1, point2):
    # Step 1: Translate point1 to the origin
    translation_matrix_1 = np.array([[1, 0, -point1[0]],
                                     [0, 1, -point1[1]],
                                     [0, 0, 1]])

    # Calculate the new coordinates of point2 after translation
    translated_point2 = np.dot(translation_matrix_1, np.array([point2[0], point2[1], 1]))

    # Step 2: Scale so that the distance between the points is 1
    distance = np.linalg.norm(translated_point2[:2])
    #     print(distance)

    scale_matrix = np.array([[1 / distance, 0, 0],
                             [0, 1 / distance, 0],
                             [0, 0, 1]])
    # Step 3: Rotate to align the line connecting the points with the vertical axis
    angle = np.arctan2(translated_point2[1], translated_point2[0]) + np.pi / 2

    #     print(angle* 180 / np.pi)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    #     print(rotation_matrix)
    # Step 4: Translate point2 to (0, 0.5)
    translation_matrix_2 = np.array([[1, 0, 0],
                                     [0, 1, 0.5],
                                     [0, 0, 1]])

    # Combine all transformations
    final_transform_matrix = np.dot(translation_matrix_2,
                                    np.dot(scale_matrix,
                                           np.dot(np.linalg.inv(rotation_matrix),
                                                  translation_matrix_1)))

    return final_transform_matrix


def apply_affine_transform(points, transformation_matrix):
    # Convert points to homogeneous coordinates (add a column of 1's)
    homogeneous_points = np.column_stack([points, np.ones(len(points))])
    # Apply the transformation matrix
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Convert back to Cartesian coordinates (remove the last column)
    transformed_points = transformed_points[:, :-1]

    return transformed_points


def apply_reverse_affine_transform(transformed_points, transformation_matrix):
    # Convert transformed points to homogeneous coordinates (add a column of 1's)
    homogeneous_transformed_points = np.column_stack([transformed_points, np.ones(len(transformed_points))])

    # Calculate the inverse of the transformation matrix
    inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

    # Apply the inverse transformation to get back to the original points
    original_points = np.dot(homogeneous_transformed_points, inverse_transformation_matrix.T)

    # Convert back to Cartesian coordinates (remove the last column)
    original_points = original_points[:, :-1]

    return original_points


def get_joint_inx(joint_ind, number_of_joints=8):
    assert joint_ind < number_of_joints, f'Joint number {joint_ind} out of range {number_of_joints - 1}'
    x_index = [i for i in range(number_of_joints * 2) if i != joint_ind * 2 and i != joint_ind * 2 + 1]
    y_index = [joint_ind * 2, joint_ind * 2 + 1]
    return x_index, y_index


def load_models(abs_path=''):

    models_folder_path = osp.join(abs_path,'joint_models')

    # Load models
    model_paths = glob.glob(osp.join(models_folder_path, '*.pth'))
    models = {}

    input_dim = 14
    output_dim = 2
    hidden_layers_config = [64, 64]
    dropout_prob = 0.2
    weight_decay = 1e-5

    for p in model_paths:
        joint_name = osp.split(p)[-1].split('_model')[0]
        model = JointModel(input_dim, output_dim, hidden_layers=hidden_layers_config, batch_norm=True,
                           weight_decay=weight_decay, dropout_prob=dropout_prob)
        model.eval()
        model.load_state_dict(torch.load(p))
        models[joint_name] = model

    scalers_path = osp.join(models_folder_path, 'scalers_dict.pkl')
    scalers = joblib.load(scalers_path)
    scalers_names = {NODES[a]: v for a, v in scalers.items()}
    return models, scalers_names


def predict_joint_pos(joint_set, models, scalers, anchor_joints=('Neck', 'Trunk')):
    number_of_joints = len(NODES)
    # get 2 anchor points
    anchor_inds = [NODES2IND[a] for a in anchor_joints]
    anchor1 = joint_set[:, anchor_inds[0]]
    anchor2 = joint_set[:, anchor_inds[1]]

    transformation_matrix = create_affine_transform_matrix(anchor1, anchor2)
    transformed_joint_set = apply_affine_transform(joint_set.T, transformation_matrix).T
    predicted_trans_joints = transformed_joint_set.copy()
    for joint, model in models.items():
        joint_ind = NODES2IND[joint]
        x_ind = [a for a in range(number_of_joints) if a != joint_ind]
        x = transformed_joint_set[:, x_ind]
        x = x.T.reshape(1, -1)
        model = models[joint]
        scaler = scalers[joint]

        x = scaler.transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        model.eval()
        pred = model(x).detach().numpy()
        predicted_trans_joints[:, joint_ind] = pred
    reconstructed_joint_set = apply_reverse_affine_transform(predicted_trans_joints.T, transformation_matrix)
    return predicted_trans_joints, transformed_joint_set, reconstructed_joint_set


def reconstruct_df(df_in, models, scalers, df_score=None, gamma=0.5, frames=None, anchor_joints=('Neck', 'Trunk')):

    df_pose_reconstructed = df_in.copy()
    node_names = NODES
    col_names = list(df_pose_reconstructed.columns)
    if frames is None:
        frames = range(len(df_pose_reconstructed))
    with tqdm(total=len(df_pose_reconstructed)) as pbar:
        for frame in frames:
            pbar.update()

            for pair_ind in range(1, 3):
                joint_set = np.zeros((2, 8))
                probs = np.zeros((1, 8))



                # get set
                if df_score is None:
                    joint_prob = 0.5
                else:
                    anchor_joints_col = [f'{anch_node}_{pair_ind}_p' for anch_node in anchor_joints]
                    joint_prob = df_score.loc[frame, anchor_joints_col].values
                    joint_prob = np.mean(joint_prob)

                for node in node_names:
                    indices = [f'{node}_{pair_ind}_x', f'{node}_{pair_ind}_y']
                    if df_score is None:
                        probs[:, NODES2IND[node]] = 0.5
                    else:
                        prob_col_name = f'{node}_{pair_ind}_p'
                        probs[:, NODES2IND[node]] = df_score.loc[frame, prob_col_name]


                    joint_set[:, NODES2IND[node]] = df_pose_reconstructed.loc[
                        frame, indices].values
                # transform set
                _, _, reconstructed_joint_set = predict_joint_pos(joint_set, models, scalers)
                reconstructed_joint_set = reconstructed_joint_set.T  # to match input

                prob_to_keep_old_joint = probs * gamma * (1-joint_prob)

                mean_values = (1 - prob_to_keep_old_joint) * reconstructed_joint_set\
                              + prob_to_keep_old_joint * joint_set
                #         mean_values = (1-gamma) * reconstructed_joint_set + gamma * joint_set

                # assign in table
                for node in node_names:
                    indice_names = [f'{node}_{pair_ind}_x', f'{node}_{pair_ind}_y']
                    indices = [col_names.index(a) for a in indice_names]

                    df_pose_reconstructed.iloc[frame, indices] = mean_values[:, NODES2IND[node]]
    return df_pose_reconstructed

if __name__ == "__main__":
    # Load models
    models, scalers = load_models()

    # joint_set = np.array(
    #     [[572.507, 664.389],
    #      [620.819, 617.056],
    #      [603.427, 720.949],
    #      [572.124, 677.219],
    #      [386.465, 408.959],
    #      [328.879, 417.663],
    #      [561.692, 578.136],
    #      [660.313, 728.908]]).T

    # anchor1 =
    # anchor2 =
    # transformation_matrix = create_affine_transform_matrix(anchor1, anchor2)
    # transformed_joint_set = apply_affine_transform(joint_set.T, transformation_matrix).T
    # transformed =
    # predicted_trans_joints, transformed_joint_set, reconstructed_joint_set = predict_joint_pos(joint_set, models,
    #                                                                                            scalers, anchor_joints=(
    #     'Neck', 'Trunk'))
    # print(predicted_trans_joints)
    # print(transformed_joint_set)
    # print(reconstructed_joint_set)
    # print('Done')
    pose_data_folder = osp.join('..','correct_simaba_files','project', 'project_folder', 'csv', 'features_extracted')
    # video_folder = osp.join('..','correct_simaba_files','cropped_videos', 'videos')

    all_video_files = glob.glob(osp.join(pose_data_folder, '*.csv'))
    for f_ind, v_files in enumerate(all_video_files):
        print(f'File number {f_ind} of {len(all_video_files)}')
        filename = osp.split(v_files)[-1]

        filename = filename.replace('.avi', '.csv')

        df_path = osp.join(pose_data_folder, filename)
        df = pd.read_csv(df_path)
        df = fix_column_names(df)

        all_joints = [a for a in df.columns if any(substring in a for substring in ['_x', '_y'])]
        score_joints = [a.replace('_x', '_p') for a in all_joints if '_x' in a]
        df_pose = df[all_joints]
        df_score = df[score_joints]

        df_pose_reconstructed = reconstruct_df(df, models, scalers, df_score=df_score, gamma=0.5, frames=None)
        df_smoothed = apply_savgol_filter(df_pose_reconstructed)
        save_file_name = osp.join('..','correct_simaba_files','reconstructed_csv', filename.replace('.csv','_reconstructed.csv'))
        df_smoothed.to_csv(save_file_name)

