# Human Pose Estimation from Depth Images
Human pose estimation from a single depth image.

# File Structure
depth-pose-estimation/
    data/
        datasets/
            CAD-60/
            ...
        processed/
            CAD-60/
                depth_images.npy
                joints.npy
            ...
    models/
        random-tree-walks/
            rtw.py
            helper.py
        ...
    output/
        random-tree-walks/
            models/
            preds/
            png/
