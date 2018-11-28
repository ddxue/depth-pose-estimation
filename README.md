# Human Pose Estimation from Depth Images
Human pose estimation from a single depth image.

# Quick-Start
(1) Process the CAD-60 dataset from data/datasets/ into numpy arrays in data/processed/.
        
        python data/process_cad_60.py
   
(2) Train network using random tree walks (RTW).
    
        python models/random-tree-walks/rtw.py
        
(3) View the visualizations in output/random-tree-walks/png/

# File Structure
    depth-pose-estimation/
        data/
            process_cad_60.py
            datasets/                   # raw dataset files
                CAD-60/
                ...
            processed/                  # processed numpy files
                CAD-60/
                    depth_images.npy    # depth images (N x H x W)
                    joints.npy          # joint coordinates (N x NUM_JOINTS x 3)
                ...
        models/
            random-tree-walks/          
                rtw.py              
                helper.py
            ...
        output/
            random-tree-walks/
                models/                 # saved models
                preds/                  # saved prediction
                png/                    # visualizations
