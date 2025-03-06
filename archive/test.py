import torch
from torchvision.models import vgg16_bn
import torchvision.transforms as T
from PIL import Image
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, get_output_shapes
from crp.visualization import FeatureVisualization
from crp.image import vis_opaque_img, plot_grid
import matplotlib.pyplot as plt
import torchvision
import os
import numpy as np

# Configuration du device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Chargement du modèle
model = vgg16_bn(weights='IMAGENET1K_V1').to(device)
model.eval()

# Préparer les transformations
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Charger ImageNet
data_path = "ImageNet_data"
imagenet_data = torchvision.datasets.ImageNet(data_path, transform=transform, split="val")
print(f"Dataset chargé avec {len(imagenet_data)} images")

# Setup pour l'analyse
cc = ChannelConcept()
composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
attribution = CondAttribution(model)

# Setup pour la visualisation
layer_names = get_layer_names(model, [torch.nn.Conv2d])
layer_map = {layer: cc for layer in layer_names}

fv = FeatureVisualization(attribution, imagenet_data, layer_map, 
                         preprocess_fn=transform.transforms[-1],  # Juste la normalisation
                         path="feature_viz")

print("Lancement de l'analyse initiale...")
# Analyser seulement la dernière couche conv pour gagner du temps
layer_id_map = {"features.40": range(512)}  # VGG16 a 512 filtres dans sa dernière couche conv

# Lancer l'analyse (cette étape peut prendre du temps)
saved_files = fv.run(composite, 0, min(1000, len(imagenet_data)), 32, 100)
print("Analyse terminée!")

def analyze_important_features(model, image, target_class, fv, threshold=0.5):
    device = next(model.parameters()).device
    image = image.clone().detach().to(device)
    image.requires_grad = True
    
    # Calculer les attributions
    conditions = [{"y": [target_class]}]
    attr = attribution(image, conditions, composite, record_layer=["features.40"])
    
    # Calculer l'importance des features
    rel_c = cc.attribute(attr.relevances["features.40"], abs_norm=True)
    importance_values, feature_indices = torch.sort(rel_c[0], descending=True)
    
    # Ne garder que les features avec une contribution > 2%
    significant_mask = torch.abs(importance_values) > 0.02
    top_features = feature_indices[significant_mask]
    top_values = importance_values[significant_mask]
    cumsum = torch.cumsum(top_values, dim=0)
    
    # Créer un dossier pour les visualisations
    viz_dir = 'feature_visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    
    plt.switch_backend('agg')
    
    print(f"\nTop features explaining {cumsum[-1]*100:.1f}% of the result:")
    for i, (feature_id, value) in enumerate(zip(top_features, top_values)):
        print(f"Feature {feature_id.item()}: {value.item()*100:.1f}%")
        
        # Créer une nouvelle figure
        fig = plt.figure(figsize=(15, 3))
        
        # Obtenir les exemples d'activation
        ref_imgs = fv.get_max_reference(
            [feature_id.item()], 
            "features.40",
            "relevance", 
            (0, 4),
            rf=True,
            composite=composite,
            plot_fn=vis_opaque_img
        )
        
        # Afficher les exemples
        plot_grid(ref_imgs, figsize=(15, 3))
        plt.title(f"Feature {feature_id.item()} - Contribution: {value.item()*100:.1f}%")
        
        # Sauvegarder et fermer
        output_path = os.path.join(viz_dir, f'feature_{feature_id.item():03d}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"Saved visualization to {output_path}")
    
    return top_features, top_values, cumsum

def compare_features(model, image1, class1, image2, class2, fv):
    print("Analyzing first image...")
    features, values1, cumsum1 = analyze_important_features(model, image1, class1, fv)
    
    print("\nAnalyzing same features on second image...")
    image2 = image2.clone().detach()
    image2.requires_grad = True
    
    conditions = [{"y": [class2]}]
    attr2 = attribution(image2, conditions, composite, record_layer=["features.40"])
    rel_c2 = cc.attribute(attr2.relevances["features.40"], abs_norm=True)
    
    values2 = rel_c2[0][features]
    cumsum2 = torch.cumsum(values2, dim=0)
    
    print(f"\nMême features expliquent {cumsum2[-1]*100:.1f}% de la deuxième classe")
    for f, v in zip(features, values2):
        print(f"Feature {f.item()}: {v.item()*100:.1f}%")



# Charger deux images de classes différentes
class1_path = os.path.join(data_path, "val/n01440764")
class2_path = os.path.join(data_path, "val/n01443537")

# Prendre la première image de chaque classe
image1_path = os.path.join(class1_path, os.listdir(class1_path)[0])
image2_path = os.path.join(class2_path, os.listdir(class2_path)[0])

# Charger et préparer les images
image1 = transform(Image.open(image1_path)).unsqueeze(0).to(device)
image2 = transform(Image.open(image2_path)).unsqueeze(0).to(device)

# Lancer l'analyse comparative
compare_features(model, image1, 0, image2, 1, fv)

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

def plot_feature_comparison(features, values1, values2, class1_name="Classe 1", class2_name="Classe 2"):
    plt.figure(figsize=(15, 6))
    x = np.arange(len(features))
    width = 0.35
    
    plt.bar(x - width/2, values1.cpu(), width, label=class1_name, color='skyblue')
    plt.bar(x + width/2, values2.cpu(), width, label=class2_name, color='lightcoral')
    
    plt.xlabel('Features')
    plt.ylabel('Contribution (%)')
    plt.title('Comparaison des contributions des features entre les deux classes')
    plt.xticks(x, features.cpu(), rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_visualizations/feature_comparison.png')
    plt.close()

def plot_input_images(image1, image2, device, class1_name="Classe 1", class2_name="Classe 2"):
    plt.figure(figsize=(10, 5))
    
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        return (tensor * std + mean).detach()  # Ajout de detach()
    
    plt.subplot(1, 2, 1)
    img1 = denormalize(image1[0]).cpu().numpy()
    plt.imshow(np.transpose(img1, (1, 2, 0)))
    plt.title(class1_name)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    img2 = denormalize(image2[0]).cpu().numpy()
    plt.imshow(np.transpose(img2, (1, 2, 0)))
    plt.title(class2_name)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_visualizations/input_images.png')
    plt.close()

def display_all_visualizations(model, image1, class1, image2, class2, fv, class1_name="Classe 1", class2_name="Classe 2"):
    device = next(model.parameters()).device
    os.makedirs('feature_visualizations', exist_ok=True)
    
    print(f"Analyzing {class1_name}...")
    image1 = image1.clone().detach().to(device)
    image1.requires_grad = True
    conditions = [{"y": [class1]}]
    attr1 = attribution(image1, conditions, composite, record_layer=["features.40"])
    rel_c1 = cc.attribute(attr1.relevances["features.40"], abs_norm=True)
    
    print(f"Analyzing {class2_name}...")
    image2 = image2.clone().detach().to(device)
    image2.requires_grad = True
    conditions = [{"y": [class2]}]
    attr2 = attribution(image2, conditions, composite, record_layer=["features.40"])
    rel_c2 = cc.attribute(attr2.relevances["features.40"], abs_norm=True)
    
    values1, indices1 = torch.sort(rel_c1[0], descending=True)
    mask = torch.abs(values1) > 0.02
    top_features = indices1[mask]
    values1 = values1[mask]
    values2 = rel_c2[0][top_features]
    
    plot_input_images(image1, image2, device, class1_name, class2_name)
    plot_feature_comparison(top_features, values1*100, values2*100, class1_name, class2_name)
    
    plt.switch_backend('agg')
    for feature_id, value1, value2 in zip(top_features, values1, values2):
        print(f"Feature {feature_id.item()}: {value1.item()*100:.1f}% vs {value2.item()*100:.1f}%")
        
        plt.figure(figsize=(15, 3))
        ref_imgs = fv.get_max_reference(
            [feature_id.item()], 
            "features.40",
            "relevance", 
            (0, 4),
            rf=True,
            composite=composite,
            plot_fn=vis_opaque_img
        )
        plot_grid(ref_imgs, figsize=(15, 3))
        plt.title(f"Feature {feature_id.item()} - Contributions: {class1_name}={value1.item()*100:.1f}%, {class2_name}={value2.item()*100:.1f}%")
        plt.savefig(f'feature_visualizations/feature_{feature_id.item():03d}.png', bbox_inches='tight', dpi=150)
        plt.close()

    print("\nVisualisations sauvegardées dans le dossier 'feature_visualizations'")
    print("- input_images.png : Images d'entrée")
    print("- feature_comparison.png : Graphique comparatif des contributions")
    print("- feature_XXX.png : Visualisations des features importantes")

# Utilisation
class1_name = "Tench"
class2_name = "Goldfish"

display_all_visualizations(model, image1, 0, image2, 1, fv, class1_name, class2_name)