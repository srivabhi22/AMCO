import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import cv2
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def preprocess_force(n,t, data):
    if t < n:
        t = n
        
    net_mean_force = np.mean(data[t-n:t])

    delta1 = net_mean_force-data[t]
    delta2 = net_mean_force-data[t]
    delta3 = net_mean_force-data[t]
    delta4 = net_mean_force-data[t]

    counter = 0

    for delta in [delta1, delta2, delta3, delta4]:
        if delta < 0:
            counter += 1
    
    delta_sum = delta1+delta2+delta3+delta4

    return [delta1, delta2, delta3, delta4, delta_sum, counter]


def preprocess_position(m, t, data_aa, data_fe):
    if t < m:
        t = m
    Ω_aa = 0
    Ω_aa += 4*(np.abs(np.max(data_aa[t-m:t])-np.min(data_aa[t-m:t])))
    Ω_fe = 0
    Ω_fe += 4*(np.abs(np.max(data_fe[t-m:t])-np.min(data_fe[t-m:t])))

    return [Ω_aa, Ω_fe]




def get_pca_components(data,t):

    scaler = StandardScaler()
    # Standardize the data
    data = scaler.fit_transform(data)
    
    pca = PCA(n_components=2)  # Reduce to 2 components for visualization

    principal_components = pca.fit_transform(data)

    # Create a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    return pca_df

# def confidence_ellipse(data, n_std=2.0, facecolor='none', **kwargs):
    
#     x, y = get_pca_components(data)

#     if x.size != y.size:
#         raise ValueError("x and y must be the same size")

#     cov = np.cov(x, y)
#     pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

#     # Calculate radii of the ellipse
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#     ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
#                       facecolor=facecolor, **kwargs)

#     # Scale radii based on the standard deviations
#     scale_x = np.sqrt(cov[0, 0]) * n_std
#     scale_y = np.sqrt(cov[1, 1]) * n_std
    
#     ellipse_area = np.pi * scale_x * scale_y
#     return ellipse, ellipse_area

def preprocess_data(path_stable_trot, path, t):
    data = pd.read_csv(path)

    print("Preprocessing Stable-Trot data........")
    preprocessed_stable_trot_data = pd.read_csv(path_stable_trot)
    resistance_amble_data = data[(data['terrain'] == 'high resistance') & (data['gait'] == 'amble')].reset_index()
    granular_crawl_data = data[(data['terrain'] == 'granular') & (data['gait'] == 'crawl')].reset_index()
    poorFoothold_amble_data = data[(data['terrain'] == 'poor foothold') & (data['gait'] == 'amble')].reset_index()


    print("Preprocessing Resistance-Amble data........")
    preprocessed_resistance_amble_data = []

    for t in tqdm(range(0, 100)):
        delta1, delta2, delta3, delta4, delta_sum, counter = preprocess_force(40, t, resistance_amble_data['knee_force'])
        Ω_aa, Ω_fe = preprocess_position(40, t, resistance_amble_data['hip_abduction_adduction'], resistance_amble_data['hip_flexion_extension'])

        preprocessed_resistance_amble_data.append([delta1, delta2, delta3, delta4, delta_sum, counter, Ω_aa, Ω_fe])

    print("Preprocessing Granular-Crawl data........")
    preprocessed_granular_crawl_data = []

    for t in tqdm(range(0, 100)):
        delta1, delta2, delta3, delta4, delta_sum, counter = preprocess_force(40, t, granular_crawl_data['knee_force'])
        Ω_aa, Ω_fe = preprocess_position(40, t, granular_crawl_data['hip_abduction_adduction'], granular_crawl_data['hip_flexion_extension'])

        preprocessed_granular_crawl_data.append([delta1, delta2, delta3, delta4, delta_sum, counter, Ω_aa, Ω_fe])

    print("Preprocessing PoorFoothold-Amble data........")
    preprocessed_poorFoothold_amble_data = []

    for t in tqdm(range(0, 100)):
        delta1, delta2, delta3, delta4, delta_sum, counter = preprocess_force(40, t, poorFoothold_amble_data['knee_force'])
        Ω_aa, Ω_fe = preprocess_position(40, t, poorFoothold_amble_data['hip_abduction_adduction'], poorFoothold_amble_data['hip_flexion_extension'])

        preprocessed_poorFoothold_amble_data.append([delta1, delta2, delta3, delta4, delta_sum, counter, Ω_aa, Ω_fe])

    
    print("Getting principal components.......")
    pca_granular_crawl = get_pca_components(preprocessed_granular_crawl_data, t)
    pca_resistance_amble = get_pca_components(preprocessed_resistance_amble_data, t)
    pca_poorFoothold_amble = get_pca_components(preprocessed_poorFoothold_amble_data, t)
    pca_stable_trot = get_pca_components(preprocessed_stable_trot_data, t)

    # _, ellipse_area_stable_trot = confidence_ellipse(preprocessed_stable_trot_data)
    # _, ellipse_area_resistance_amble = confidence_ellipse(preprocessed_resistance_amble_data)
    # _, ellipse_area_granular_crawl = confidence_ellipse(preprocessed_granular_crawl_data)
    # _, ellipse_area_poorFoothold_amble = confidence_ellipse(preprocessed_poorFoothold_amble_data)

    return pca_granular_crawl, pca_resistance_amble, pca_poorFoothold_amble, pca_stable_trot
