#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate SMPL-X body mesh and visualize from MoSh++ results
"""

import os
import os.path as osp
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add paths (Isaak so create_smplx_gif can be imported; moshpp/soma for MoSh)
_isaak_dir = osp.dirname(osp.abspath(__file__))
if _isaak_dir not in sys.path:
    sys.path.insert(0, _isaak_dir)
sys.path.insert(0, osp.join(osp.dirname(__file__), '../../moshpp/src'))
sys.path.insert(0, osp.join(osp.dirname(__file__), '../../soma/src'))

try:
    from smplx import SMPLX
    import torch
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False

from loguru import logger

try:
    from human_body_prior.body_model.body_model import BodyModel
    # c2c is a utility function - try to find it
    try:
        from human_body_prior.tools.omni_tools import c2c
    except:
        # Alternative location
        def c2c(x):
            if hasattr(x, 'detach'):
                return x.detach().cpu().numpy()
            return np.array(x)
    BODY_MODEL_AVAILABLE = True
except ImportError as e:
    BODY_MODEL_AVAILABLE = False
    logger.warning(f"BodyModel not available: {e}")

def load_smplx_model(model_path, gender='neutral'):
    """Load SMPL-X model"""
    if SMPLX_AVAILABLE:
        model = SMPLX(model_path=osp.dirname(model_path),
                     gender=gender,
                     use_face_contour=False,
                     num_betas=10,  # Use first 10 shape parameters
                     ext='pkl')
        return model
    else:
        # Fallback: load from pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f, encoding='latin-1')
        return model_data

def generate_body_mesh_smplx(model, betas, pose, trans=None):
    """Generate body mesh from SMPL-X parameters"""
    if SMPLX_AVAILABLE and isinstance(model, SMPLX):
        # Use SMPL-X library
        betas_tensor = torch.from_numpy(betas[:10]).float().unsqueeze(0)
        pose_tensor = torch.from_numpy(pose).float().unsqueeze(0)
        
        if trans is not None:
            trans_tensor = torch.from_numpy(trans).float().unsqueeze(0)
        else:
            trans_tensor = torch.zeros(1, 3)
        
        # Generate mesh
        output = model(betas=betas_tensor, 
                      body_pose=pose_tensor[:, 3:66],  # Body pose (21 joints * 3)
                      global_orient=pose_tensor[:, :3],  # Global rotation
                      transl=trans_tensor)
        
        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = model.faces
        return vertices, faces
    else:
        # Fallback: use MoSh++ body model loader
        from moshpp.bodymodel_loader import load_moshpp_models
        from moshpp.tools.run_tools import setup_mosh_omegaconf_resolvers
        
        setup_mosh_omegaconf_resolvers()
        
        # Load model using MoSh++ loader
        surface_model = load_moshpp_models(
            surface_model_fname=model_path,
            surface_model_type='smplx',
            surface_model_gender='neutral'
        )
        
        # Use the model to generate vertices
        # This is a simplified version - MoSh++ has its own way of generating meshes
        logger.warning("Using MoSh++ model loader - mesh generation may be limited")
        return None, None

def visualize_body_mesh(vertices, faces, frame_idx=0, save_path=None):
    """
    Render one frame of the body mesh using pyrender (EGL, Z-buffered).

    Vertices stay in native SMPL-X Y-up world space (no rotation needed).
    Camera is placed at +Z in front of the person (who faces +Z in SMPL-X)
    looking back toward the centroid with Y as up.
    """
    import os
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh as _trimesh

    def _look_at(eye, target, up=None):
        if up is None:
            up = np.array([0., 1., 0.])
        fwd = target - eye; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, up); right /= np.linalg.norm(right)
        tup = np.cross(right, fwd)
        m = np.eye(4, dtype=np.float64)
        m[:3,0]=right; m[:3,1]=tup; m[:3,2]=-fwd; m[:3,3]=eye
        return m

    tri = _trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.65, 0.74, 0.86, 1.0],
        metallicFactor=0.0, roughnessFactor=0.6, smooth=True)
    py_mesh = pyrender.Mesh.from_trimesh(tri, material=mat, smooth=True)

    centroid = vertices.mean(axis=0)
    body_h = vertices[:, 1].max() - vertices[:, 1].min()  # Y is height
    dist = max(2.5, body_h * 1.6)

    bg_env = os.environ.get("SMPLX_BG_COLOR", "").strip()
    if bg_env:
        try:
            bg_vals = [float(x.strip()) for x in bg_env.split(",")]
            if len(bg_vals) == 3:
                bg_color = [bg_vals[0], bg_vals[1], bg_vals[2], 1.0]
            elif len(bg_vals) == 4:
                bg_color = bg_vals
            else:
                bg_color = [0.95, 0.95, 0.95, 1.0]
        except Exception:
            bg_color = [0.95, 0.95, 0.95, 1.0]
    else:
        bg_color = [0.95, 0.95, 0.95, 1.0]

    scene = pyrender.Scene(bg_color=bg_color,
                           ambient_light=[0.35, 0.35, 0.35])
    scene.add(py_mesh)

    key_pose  = _look_at(centroid + np.array([dist*0.6,  body_h*0.4,  dist]),     centroid)
    fill_pose = _look_at(centroid + np.array([-dist*0.5, body_h*0.1,  dist*0.8]), centroid)
    scene.add(pyrender.DirectionalLight(color=[1.,1.,1.], intensity=4.0), pose=key_pose)
    scene.add(pyrender.DirectionalLight(color=[0.9,0.9,1.], intensity=2.0), pose=fill_pose)

    cam_eye = centroid + np.array([0., body_h * 0.05, dist * 1.05])
    cam_pose = _look_at(eye=cam_eye, target=centroid)
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(42), znear=0.05, zfar=50.), pose=cam_pose)

    r = pyrender.OffscreenRenderer(640, 720)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
    r.delete()

    if save_path:
        from PIL import Image
        Image.fromarray(color).save(save_path)
        logger.info(f"Saved visualization to {save_path}")
    else:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 7))
        plt.imshow(color)
        plt.axis('off')
        plt.title(f'SMPL-X Body Mesh - Frame {frame_idx}')
        plt.tight_layout()
        plt.show()

    return color


def visualize_pose_sequence(poses, trans, save_path=None, num_frames=10):
    """Visualize pose sequence as skeleton"""
    fig = plt.figure(figsize=(15, 10))
    
    # Sample frames
    frame_indices = np.linspace(0, len(poses)-1, num_frames, dtype=int)
    
    for idx, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(2, 5, idx+1, projection='3d')
        
        # Extract root position
        root_pos = trans[frame_idx] if trans is not None else np.zeros(3)
        
        # For visualization, we'll plot the root position and indicate pose complexity
        # Full pose visualization would require joint hierarchy.
        # SMPL-X Y-up → matplotlib Z-up: pass as (X, Z, Y).
        ax.scatter(root_pos[0], root_pos[2], root_pos[1],
                  c='red', s=100, marker='o')
        
        # Plot pose parameters as a simplified representation
        pose_params = poses[frame_idx]
        # Reshape pose to (num_joints, 3) - approximate
        num_joints = len(pose_params) // 3
        pose_reshaped = pose_params[:num_joints*3].reshape(num_joints, 3)
        
        # Plot first few joints relative to root
        for i, joint_rot in enumerate(pose_reshaped[:10]):  # First 10 joints
            # Simplified: show joint position based on rotation
            joint_pos = root_pos + joint_rot * 0.1  # Scale for visualization
            ax.scatter(joint_pos[0], joint_pos[2], joint_pos[1],
                      c='blue', s=20, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y - Height ↑')
        ax.set_title(f'Frame {frame_idx}')
        
        # Set equal aspect
        max_range = 1.0
        ax.set_xlim(root_pos[0] - max_range, root_pos[0] + max_range)
        ax.set_ylim(root_pos[2] - max_range, root_pos[2] + max_range)
        ax.set_zlim(root_pos[1] - max_range, root_pos[1] + max_range)
    
    plt.suptitle('SMPL-X Pose Sequence Visualization', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved pose sequence visualization to {save_path}")
    else:
        plt.show()
    
    return fig

def main():
    """Main function"""
    import os
    # Paths for the 24-marker MoSh++ run (override with env STAGEII, OUT_DIR, SUPPORT_BASE_DIR)
    stageii_path = os.environ.get(
        'STAGEII',
        '/home/isaak/BEP/Isaak/output_mosh24/markers24_labeled_stageii.pkl'
    )
    model_path = os.environ.get(
        'SMPLX_MODEL',
        '/home/isaak/BEP/body_models/smplx/SMPLX_NEUTRAL.pkl'
    )
    output_dir = os.environ.get(
        'OUT_DIR',
        '/home/isaak/BEP/Isaak/output_mosh24'
    )
    support_base_dir = os.environ.get(
        'SUPPORT_BASE_DIR',
        '/home/isaak/BEP/moshpp/support_data'
    )
    
    logger.info("Loading MoSh++ results...")
    
    # Load directly from pickle to preserve all 275 frames
    # (MoSh.load_as_amass_npz silently drops the last frame)
    with open(stageii_path, 'rb') as f:
        mosh_result_raw = pickle.load(f)
    try:
        from moshpp.mosh_head import MoSh
        mosh_result = MoSh.load_as_amass_npz(stageii_path, include_markers=True)
        # Restore full-length trans/fullpose from raw pkl
        mosh_result['trans'] = np.array(mosh_result_raw['trans'])
        mosh_result['fullpose'] = np.array(mosh_result_raw['fullpose'])
        logger.info("Loaded using MoSh.load_as_amass_npz (trans/poses restored from raw pkl)")
    except Exception as e:
        logger.warning(f"MoSh.load_as_amass_npz failed: {e}, using raw pkl directly")
        mosh_result = mosh_result_raw
    
    betas = mosh_result['betas']
    poses = mosh_result.get('fullpose', None)
    if poses is None:
        # Try to reconstruct from parts
        poses = mosh_result.get('poses', None)
    trans = mosh_result['trans']
    
    logger.info(f"Loaded {len(trans)} frames")
    logger.info(f"Betas shape: {betas.shape if hasattr(betas, 'shape') else 'scalar'}")
    if poses is not None:
        logger.info(f"Poses shape: {poses.shape}")
    logger.info(f"Trans shape: {trans.shape}")
    
    # Generate mesh using SOMA's produce_body_from_mosh_pkl function
    vertices, faces = None, None
    
    logger.info(f"BODY_MODEL_AVAILABLE: {BODY_MODEL_AVAILABLE}")
    
    if BODY_MODEL_AVAILABLE:
        try:
            logger.info("Generating SMPL-X body mesh using SOMA's method...")
            logger.info(f"Using support_base_dir: {support_base_dir}")
            
            # Use SOMA's function directly, but modify to use PKL instead of NPZ
            from soma.tools.eval_v2v import produce_body_from_mosh_pkl
            import os.path as osp
            
            # Temporarily create a symlink or copy PKL to NPZ location if needed
            model_pkl_path = osp.join(support_base_dir, 'smplx', 'neutral', 'model.pkl')
            model_npz_path = osp.join(support_base_dir, 'smplx', 'neutral', 'model.npz')
            
            # If NPZ doesn't work, try using PKL directly by modifying the function call
            # For now, let's try the function and catch the error
            try:
                body_result = produce_body_from_mosh_pkl(stageii_path, support_base_dir)
            except ValueError as e:
                if 'allow_pickle' in str(e):
                    # BodyModel needs PKL format, let's use PKL directly
                    logger.info("NPZ format issue, trying to use PKL directly...")
                    # Modify the function to use PKL
                    import sys
                    sys.path.insert(0, osp.join(osp.dirname(__file__), '../../soma/src'))
                    from moshpp.mosh_head import MoSh
                    from human_body_prior.body_model.body_model import BodyModel
                    from human_body_prior.tools.omni_tools import copy2cpu as c2c
                    import torch
                    
                    mosh_result = MoSh.load_as_amass_npz(stageii_path, include_markers=True)
                    surface_model_type = mosh_result.get('surface_model_type', 'smplx')
                    gender = mosh_result.get('gender', 'neutral')
                    
                    # Use PKL file directly
                    surface_model_fname = model_pkl_path
                    if not osp.exists(surface_model_fname):
                        surface_model_fname = model_path
                    
                    logger.info(f"Loading BodyModel from {surface_model_fname}")
                    sm = BodyModel(bm_fname=surface_model_fname,
                                 num_betas=mosh_result.get('num_betas', 10),
                                 num_expressions=mosh_result.get('num_expressions', 0),
                                 num_dmpls=None,
                                 dmpl_fname=None)
                    
                    body_result = {}
                    body_result['surface_f'] = c2c(sm.f)
                    
                    time_length = len(mosh_result['trans'])
                    selected_frames = range(0, time_length)
                    
                    if 'betas' in mosh_result:
                        mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=time_length, axis=0)
                    
                    body_keys = ['betas', 'trans', 'pose_body', 'root_orient', 'pose_hand']
                    if surface_model_type == 'smplx':
                        body_keys += ['expression']
                    
                    surface_parms = {k: torch.Tensor(v[selected_frames]) for k, v in mosh_result.items() if k in body_keys}
                    body_result['surface_v'] = c2c(sm(**surface_parms).v)
                    body_result['betas'] = mosh_result['betas']
                else:
                    raise
            
            logger.info(f"Body result keys: {list(body_result.keys())[:15]}")
            
            # BodyModel may output in model space (no global trans). Add root translation so mesh is in world space (head ~1.8 m like your markers).
            if 'surface_v' in body_result and 'trans' in body_result:
                trans_arr = body_result['trans']
                if hasattr(trans_arr, 'shape') and len(trans_arr.shape) >= 2:
                    body_result['surface_v'] = body_result['surface_v'] + trans_arr[:, np.newaxis, :]
                    logger.info("Applied root translation to mesh (world space)")
            
            # Extract mesh for first frame
            if 'surface_v' in body_result and 'surface_f' in body_result:
                vertices = body_result['surface_v'][0]  # First frame
                faces = body_result['surface_f']
                
                logger.info(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
                
                # Visualize
                vis_path = osp.join(output_dir, 'smplx_body_mesh_frame0.png')
                visualize_body_mesh(vertices, faces, frame_idx=0, save_path=vis_path)
                
                # Save mesh data for first frame
                mesh_data = {
                    'vertices': vertices,
                    'faces': faces,
                    'betas': body_result.get('betas', betas),
                    'frame_0_pose': poses[0] if poses is not None else None,
                    'frame_0_trans': trans[0]
                }
                mesh_path = osp.join(output_dir, 'smplx_body_mesh_frame0.pkl')
                with open(mesh_path, 'wb') as f:
                    pickle.dump(mesh_data, f)
                logger.info(f"Saved mesh data to {mesh_path}")
                
                # Also save all frames mesh
                all_frames_mesh_path = osp.join(output_dir, 'smplx_body_mesh_all_frames.pkl')
                with open(all_frames_mesh_path, 'wb') as f:
                    pickle.dump({
                        'vertices': body_result['surface_v'],  # All frames
                        'faces': faces,
                        'betas': body_result.get('betas', betas),
                        'poses': poses,
                        'trans': trans
                    }, f)
                logger.info(f"Saved all frames mesh data to {all_frames_mesh_path}")
                # Create GIF using same code as Isaak/create_smplx_gif.py (run same way as zzcommands)
                gif_path = osp.join(output_dir, 'smplx_body_animation.gif')
                try:
                    from create_smplx_gif import create_gif_from_mesh
                    create_gif_from_mesh(
                        all_frames_mesh_path, gif_path,
                        fps=15, max_frames=None, downsample=1
                    )
                except Exception as e:
                    logger.warning(f"GIF creation failed: {e}")
            else:
                logger.warning(f"Mesh generation did not produce surface_v or surface_f. Available keys: {list(body_result.keys())}")
                
        except Exception as e:
            logger.error(f"SOMA method failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Create pose sequence visualization
    if poses is not None:
        logger.info("Creating pose sequence visualization...")
        pose_seq_path = osp.join(output_dir, 'smplx_pose_sequence.png')
        visualize_pose_sequence(poses, trans, save_path=pose_seq_path, num_frames=10)
    
    logger.success("Visualization complete!")

if __name__ == '__main__':
    main()
