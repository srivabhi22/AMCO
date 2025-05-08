import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import cv2
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from src.preprocessing.preprocess_data import preprocess_data
from src.preprocessing.haar_wavelet import get_blur_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def preprocess_mask(seg_image_at_time_t):
    """
    Crop the mask slightly more than the middle and resize it back to the original size.
    """

    
    num_rows, num_cols = seg_image_at_time_t.shape[:2]
    crop_start = int(0.55 * num_rows)  # Cropping slightly more than the middle
    cropped_mask = seg_image_at_time_t[crop_start:, :]
    
    # Resize back to original dimensions
    resized_mask = cv2.resize(cropped_mask, (num_cols, num_rows), interpolation=cv2.INTER_NEAREST)
    return resized_mask

def assign_labels(rgb_mask, group_colors):
        
        
        label_mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
        
        for i in range(rgb_mask.shape[0]):
            for j in range(rgb_mask.shape[1]):
                pixel = rgb_mask[i, j]
                found = False
                for key, value in group_colors.items():
                    if np.array_equal(pixel, value):
                        label_mask[i, j] = key
                        found = True
                        break
                if not found:
                    label_mask[i, j] = 0  # Default label for undefined colors

        return label_mask

def current_terrain(mask):
    # Define the bottom third of the image
    h, w = mask.shape
    bottom_third = mask[int(h * 2/3):, :]

    # Count occurrences of only valid labels (1, 2, 3)
    valid_labels = bottom_third.flatten()
    valid_labels = valid_labels[np.isin(valid_labels, [1, 2, 3])]

    if len(valid_labels) > 0:
        label_counts = Counter(valid_labels)
        selected_label = max(label_counts, key=label_counts.get)  # Select the most frequent label
    else:
        # Expand search to the middle third if no valid labels were found
        expanded_search = mask[int(h * 1/2):, :]
        expanded_labels = expanded_search.flatten()
        expanded_labels = expanded_labels[np.isin(expanded_labels, [1, 2, 3])]

        if len(expanded_labels) > 0:
            label_counts = Counter(expanded_labels)
            selected_label = max(label_counts, key=label_counts.get)
        else:
            selected_label = None  # No valid terrain detected
    
    return selected_label




def general_knowledge_cost_map(n, seg_image_at_time_t, gamma, ellipse_area_stable_trot, 
                               ellipse_area_resistance_amble, ellipse_area_granular_crawl, 
                               ellipse_area_poorFoothold_amble):
    # Calculate the number of grids along each dimension
    num_rows, num_cols = seg_image_at_time_t.shape
    num_grids_row = (num_rows + n - 1) // n  # Include partial grids
    num_grids_col = (num_cols + n - 1) // n  # Include partial grids

    # Create a new array to store the modified values
    updated_mask = np.zeros_like(seg_image_at_time_t, dtype=float)

    # Iterate over the grids
    for i in range(num_grids_row):
        for j in range(num_grids_col):
            # Determine the start and end indices for the current grid
            start_row = i * n
            end_row = min(start_row + n, num_rows)  # Ensure within bounds
            start_col = j * n
            end_col = min(start_col + n, num_cols)  # Ensure within bounds

            # Extract the current grid
            grid = seg_image_at_time_t[start_row:end_row, start_col:end_col]
            
            # Flatten the grid and count frequencies
            flattened_grid = grid.flatten()
            category_counts = Counter(flattened_grid)
            
            # Find the category with the maximum frequency
            max_category = max(category_counts, key=category_counts.get)
            
            # Calculate the value based on the category
            if max_category == 1:
                value = (gamma / n**2) * (ellipse_area_stable_trot)
            elif max_category == 2:
                value = (gamma / n**2) * (ellipse_area_granular_crawl)
            elif max_category == 3:
                value = (gamma / n**2) * (ellipse_area_poorFoothold_amble)
            elif max_category == 4:
                value = (gamma / n**2) * (ellipse_area_resistance_amble)
            else:
                value = 1e6  # Optional: Handle case if the max category is 0 or undefined
            
            # Assign the `value` to all pixels in the current grid
            updated_mask[start_row:end_row, start_col:end_col] = value

    return updated_mask
 


def traversibility_history_cost_map(n, t, seg_image_at_time_t, alpha, terrain_history,
                               l2_norm_granular_crawl, 
                               l2_norm_resistance_amble, 
                               l2_norm_poorFoothold_amble, 
                               l2_norm_stable_trot,
                               general_knowledge_map,
                               past_terrains):
    # Calculate the number of grids along each dimension
    num_rows, num_cols = seg_image_at_time_t.shape
    num_grids_row = (num_rows + n - 1) // n  # Include partial grids
    num_grids_col = (num_cols + n - 1) // n  # Include partial grids

    # Create a new array to store the modified values
    updated_mask = np.zeros_like(seg_image_at_time_t, dtype=float)

    # Iterate over the grids
    for i in range(num_grids_row):
        for j in range(num_grids_col):
            # Determine the start and end indices for the current grid
            start_row = i * n
            end_row = min(start_row + n, num_rows)  # Ensure within bounds
            start_col = j * n
            end_col = min(start_col + n, num_cols)  # Ensure within bounds

            # Extract the current grid
            grid = seg_image_at_time_t[start_row:end_row, start_col:end_col]
            
            # Flatten the grid and count frequencies
            flattened_grid = grid.flatten()
            category_counts = Counter(flattened_grid)
            
            # Find the category with the maximum frequency
            max_category = max(category_counts, key=category_counts.get)
            
            # Calculate the value based on the category
            if max_category == 1:
                if 'granular' in past_terrains:
                    value = alpha*(terrain_history['granular']*(l2_norm_granular_crawl-general_knowledge_map[start_row:end_row, start_col:end_col]))
                else:
                    value = 0

            elif max_category == 2:
                if 'resistance' in past_terrains:
                    value = alpha*(terrain_history['resistance']*(l2_norm_resistance_amble-general_knowledge_map[start_row:end_row, start_col:end_col]))
                else:
                    value = 0

            elif max_category == 3 and 'stable' in past_terrains:
                if 'stable' in past_terrains:
                    value = alpha*(terrain_history['stable']*(l2_norm_stable_trot-general_knowledge_map[start_row:end_row, start_col:end_col]))
                else:
                    value = 0

            elif max_category == 4 and 'poorFoothold' in past_terrains:
                if 'poorFoothold' in past_terrains:
                    value = alpha*(terrain_history['poorFoothold']*(l2_norm_poorFoothold_amble-general_knowledge_map[start_row:end_row, start_col:end_col]))
                else:
                    value = 0
            else:
                value = 1e6  # Optional: Handle case if the max category is 0 or undefined
            
            # Assign the `value` to all pixels in the current grid
            updated_mask[start_row:end_row, start_col:end_col] = value

    return updated_mask

def current_proprioceptivity_cost_map(seg_image, t, µ, U, n, 
                                 pca_component_time_t):
    # Calculate the number of grids along each dimension
    num_rows, num_cols = seg_image.shape
    num_grids_row = (num_rows + n - 1) // n  # Include partial grids
    num_grids_col = (num_cols + n - 1) // n  # Include partial grids

    # Create a new array to store the modified values
    updated_mask = np.zeros_like(seg_image, dtype=float)
    
    # Bottom center grid index for distance calculation
    i_bottom_center = num_grids_row - 1
    j_bottom_center = num_grids_col // 2
    d_max = np.sqrt((num_grids_row - 1)**2 + (num_grids_col - 1)**2)

    # Iterate over the grids
    for i in range(num_grids_row):
        for j in range(num_grids_col):
            # Determine the start and end indices for the current grid
            start_row = i * n
            end_row = min(start_row + n, num_rows)  # Ensure within bounds
            start_col = j * n
            end_col = min(start_col + n, num_cols)  # Ensure within bounds

            # Extract the current grid
            d_ij = np.sqrt((i - i_bottom_center)**2 + (j - j_bottom_center)**2)
            normalized_distance = d_ij / d_max

            value = (U - µ * pca_component_time_t) * (1 - normalized_distance)
            updated_mask[start_row:end_row, start_col:end_col] = value

    return updated_mask


def coupled_map(image_path, mask_path, t=100):


    ellipse_area_stable_trot = 28.14
    ellipse_area_resistance_amble = 29.96
    ellipse_area_granular_crawl = 29.95
    ellipse_area_poorFoothold_amble = 29.99

    n = 40
    µ = 31.875
    U = 127
    alpha = 4.5
    gamma = 1  
    terrain_history = {'granular': 131.0, 'resistance': 30.0, 'stable': 126.0, 'poorFoothold': 180.0}
    past_terrains = ['granular', 'stable']

    # 0 -- Background: void, sky, sign
    # 1 -- Level1 (smooth) - Navigable: concrete, asphalt
    # 2 -- Level2 (rough) - Navigable: gravel, grass, dirt, sand, mulch
    # 3 -- Level3 (bumpy) - Navigable: Rock, Rock-bed
    # 4 -- Non-Navigable (forbidden) - water
    # 5 -- Obstacle - tree, pole, vehicle, container/generic-object, building, log, 
    #                 bicycle(could be removed), person, fence, bush, picnic-table, bridge,

    GROUP_COLORS = { # have beein defined according to RGB color space
            0: [0, 0, 0],        # sky
            1: [0, 128, 0],      # stable
            2: [255, 255, 0],    # granular
            3: [255, 128, 0],    # poor foothold
            4: [0, 0, 255],      # forbidden
            5: [255, 0, 0]       # obstacle
    }

    print("Reading image.......")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_mask(image)

    print("Getting Segmented Mask.......")
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = preprocess_mask(mask)
    mask_2D = assign_labels(mask, GROUP_COLORS)

   
    pca_components_granular_crawl_time_t, pca_components_resistance_amble_time_t, pca_components_poorFoothold_amble_time_t, pca_components_stable_trot_time_t = preprocess_data("/home/srivabhi22/Desktop/AMCO/data/sensor/preprocessed_stable_trot.csv",
                                                                                                                        "/home/srivabhi22/Desktop/AMCO/data/sensor/quadruped_proprioceptive_data_with_trends.csv", t)
    
    l2_norm_granular_crawl = np.sqrt(np.sum(pca_components_granular_crawl_time_t["PC1"]**2 +pca_components_granular_crawl_time_t["PC2"]**2))
    l2_norm_resistance_amble = np.sqrt(np.sum(pca_components_resistance_amble_time_t["PC1"]**2 +pca_components_resistance_amble_time_t["PC2"]**2))
    l2_norm_poorFoothold_amble = np.sqrt(np.sum(pca_components_poorFoothold_amble_time_t["PC1"]**2 +pca_components_poorFoothold_amble_time_t["PC2"]**2))
    l2_norm_stable_trot = np.sqrt(np.sum(pca_components_stable_trot_time_t["PC1"]**2 +pca_components_stable_trot_time_t["PC2"]**2))

    print("Creating General Knowledge Map.......")
    general_knowledge_map = general_knowledge_cost_map(n, mask_2D, gamma, ellipse_area_stable_trot, 
                                                        ellipse_area_resistance_amble, ellipse_area_granular_crawl, 
                                                        ellipse_area_poorFoothold_amble)
    
    np.save("/home/srivabhi22/Desktop/AMCO/data/maps/general_map.npy", general_knowledge_map)
    

    print("Creating Traversibility History Map.......")
    traversibility_history_map = traversibility_history_cost_map(n, t, mask_2D, alpha, terrain_history,
                                                                l2_norm_granular_crawl, 
                                                                l2_norm_resistance_amble, 
                                                                l2_norm_poorFoothold_amble, 
                                                                l2_norm_stable_trot,
                                                                general_knowledge_map,
                                                                past_terrains)
    np.save("/home/srivabhi22/Desktop/AMCO/data/maps/history_map.npy", traversibility_history_map)

    print("Creating Current Proprioceptive Map.......")
    # Determine the terrain the robot is standing on
    terrain_class = current_terrain(mask_2D)

    # Default value in case the terrain class does not match known types
    current_proprioceptivity_map = None

    if terrain_class == 1:
        current_proprioceptivity_map = current_proprioceptivity_cost_map(mask_2D, t, µ, U, n,  
                                                                     l2_norm_stable_trot)
    elif terrain_class == 2:
        current_proprioceptivity_map = current_proprioceptivity_cost_map(mask_2D, t, µ, U, n, 
                                                                     l2_norm_granular_crawl)
    elif terrain_class == 3:
        current_proprioceptivity_map = current_proprioceptivity_cost_map(mask_2D, t, µ, U, n, 
                                                                     l2_norm_poorFoothold_amble)
    # elif terrain_class == 4:
    #     current_proprioceptivity_map = current_proprioceptivity_cost_map(mask_2D, t, µ, U, n, 
    #                                                                  l2_norm_resistance_amble)
    else:
        print(f"Warning: Unknown terrain class {terrain_class}. Using default proprioceptive map.")
        current_proprioceptivity_map = np.zeros_like(mask_2D)  # Assign a default map

    # Ensure current_proprioceptivity_map is defined before saving
    if current_proprioceptivity_map is not None:
        np.save("/home/srivabhi22/Desktop/AMCO/data/maps/proprioceptive_map.npy", current_proprioceptivity_map)
    else:
        raise ValueError("Failed to create proprioceptive map: No valid terrain class detected.")


    
    print("Creating Coupled Traversibility Map.......")
    # Combine the three maps

    mean_brightness = np.sum(image)/(3*image.shape[0]*image.shape[1])
    weighted_brightness = np.sum(image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114)/(image.shape[0]*image.shape[1])

    haar_wavelet_score = get_blur_score(image_path)
    resnet_score = 0.98
    ξ = 0.0008*mean_brightness + 0.001*weighted_brightness + 0.0025*haar_wavelet_score + 0.003*resnet_score
    coupled_cost_map = ξ*(general_knowledge_map + traversibility_history_map) + current_proprioceptivity_map
    # cv2.imwrite("/home/srivabhi22/Desktop/AMCO/data/maps/coupled_map.png", coupled_cost_map)
    np.save("/home/srivabhi22/Desktop/AMCO/data/maps/coupled_map.npy", coupled_cost_map)
    print(f"mean brightness:{mean_brightness}, wieghted brightness:{weighted_brightness}, Blur extent:{haar_wavelet_score}, reliability:{ξ}")
    return coupled_cost_map


