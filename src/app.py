import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg16_bn
from PIL import Image
from pathlib import Path
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from crp.image import vis_opaque_img
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm
import pandas as pd
import numpy as np
import time
import os

# 🔍 GPU Check
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# Configuration des datasets
DATASET_CONFIGS = {
    "ImageNet": {
        "transform": T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ]),
        "preprocessing": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        "path": "ImageNet_data",
        "split": "val"
    },
    "CIFAR-10": {
        "transform": T.Compose([
            T.Resize(224),
            T.ToTensor()
        ]),
        "preprocessing": T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        "path": "./cifar10",
        "split": "test"
    }
}

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_dataset(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    if dataset_name == "ImageNet":
        dataset = torchvision.datasets.ImageNet(
            config["path"],
            transform=config["transform"],
            split=config["split"]
        )
    else:
        dataset = torchvision.datasets.CIFAR10(
            root=config["path"],
            train=False,
            transform=config["transform"],
            download=True
        )
    return dataset, config["preprocessing"]

@st.cache_resource
def load_model():
    model = vgg16_bn(weights="IMAGENET1K_V1").to(device)
    model.eval()
    return model

def get_class_list(dataset_name, dataset):
    if dataset_name == "ImageNet":
        return list(dataset.class_to_idx.keys())
    else:
        return CIFAR10_CLASSES

def compute_feature_importance(model, dataset, layer_idx, num_features, pred_class, dataset_name):
    sum_original_score = 0.0
    feature_importance = {}
    
    if dataset_name == "CIFAR-10":
        pred_class = CIFAR10_CLASSES.index(pred_class)
    
    # Calcul des scores originaux
    for idx in range(min(100, len(dataset))):
        input_tensor, target = dataset[idx]
        input_tensor = input_tensor.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            output_original = model(input_tensor)
            probs_original = torch.nn.functional.softmax(output_original, dim=1)
            sum_original_score += probs_original[0, pred_class].item()
    
    mean_original_score = sum_original_score / min(100, len(dataset))

    # Test de chaque feature
    for feature_idx in range(num_features):
        def zero_out_feature(module, input, output, feature_idx=feature_idx):
            output[:, feature_idx, :, :] = 0
            return output

        hook = model.features[layer_idx].register_forward_hook(zero_out_feature)
        sum_new_score = 0.0
        
        for idx in range(min(100, len(dataset))):
            input_tensor, target = dataset[idx]
            input_tensor = input_tensor.unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                output_disabled = model(input_tensor)
                probs_disabled = torch.nn.functional.softmax(output_disabled, dim=1)
                sum_new_score += probs_disabled[0, pred_class].item()
        
        mean_new_score = sum_new_score / min(100, len(dataset))
        hook.remove()
        
        importance = mean_original_score - mean_new_score
        feature_importance[feature_idx] = importance
        
    return dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

def run_extraction(fv, dataset, composite):
    st.info("🔍 Lancement de l'analyse... Cela peut prendre plusieurs minutes ⏳")
    torch.cuda.synchronize()
    start_time = time.time()
    try:
        saved_files = fv.run(composite, 0, min(1000, len(dataset)), 32, 100)
        duration = time.time() - start_time
        st.success(f"✅ Analyse terminée et sauvegardée en {duration:.2f} secondes !")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de l'exécution : {e}")

def main():
    try:
        st.set_page_config(page_title="Exploration des Features CNN", layout="wide")
    except:
        pass

    # Sélection du dataset
    dataset_name = st.sidebar.selectbox("📊 Sélection du Dataset:", ["ImageNet", "CIFAR-10"])

    # Chargement du modèle et du dataset
    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    
    dataset, preprocessing = load_dataset(dataset_name)
    model = st.session_state.model

    # Configuration CRP
    cc = ChannelConcept()
    composite = EpsilonPlusFlat([SequentialMergeBatchNorm()])
    attribution = CondAttribution(model)
    layer_names = get_layer_names(model, [torch.nn.Conv2d])
    layer_map = {layer: cc for layer in layer_names}
    
    # Gestion des chemins de cache
    base_cache_dir = Path("feature_viz_cache")
    dataset_cache_dir = base_cache_dir / dataset_name.lower()
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    features_exist = any(dataset_cache_dir.glob("*.pt"))
    
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing, path=str(dataset_cache_dir))

    # Menu latéral
    menu = st.sidebar.radio("📑 Menu :", ["⚙️ Extraction des Features", "📊 Analyse des Features Importantes", "🖼️ Visualisation"])

    # Menu Extraction
    if menu == "⚙️ Extraction des Features":
        st.title(f"⚙️ Extraction des Features - {dataset_name}")
        
        if features_exist:
            st.success("✅ Features déjà extraites pour ce dataset !")
            if st.button("🔄 Ré-extraire les features"):
                run_extraction(fv, dataset, composite)
        else:
            if st.button("🚀 Lancer l'analyse complète"):
                run_extraction(fv, dataset, composite)

    # Menu Analyse
    elif menu == "📊 Analyse des Features Importantes":
        st.title(f"📊 Analyse des Features Importantes - {dataset_name}")
        
        class_list = get_class_list(dataset_name, dataset)
        default_classes = class_list[:2] if len(class_list) > 1 else [class_list[0]]
        
        selected_classes = st.multiselect("🎯 Choisissez les classes :", class_list, default=default_classes)
        n_features = st.slider("🔢 Nombre de features à afficher :", 1, 10, 5)

        if st.button("📋 Afficher les features importantes"):
            st.info("🧠 Calcul des importances sur GPU...")
            results = []
            
            for cls in selected_classes:
                if dataset_name == "ImageNet":
                    cls_idx = dataset.class_to_idx[cls]
                    filtered_dataset = [(img, label) for img, label in dataset if label == cls_idx]
                else:
                    cls_idx = CIFAR10_CLASSES.index(cls)
                    filtered_dataset = [(img, label) for img, label in dataset if label == cls_idx]

                importance_dict = compute_feature_importance(
                    model, filtered_dataset, layer_idx=40, num_features=512,
                    pred_class=cls, dataset_name=dataset_name
                )

                for f_id, imp in list(importance_dict.items())[:n_features]:
                    results.append({"Classe": cls, "Feature": f_id, "Importance": imp})

            if results:
                df = pd.DataFrame(results)
                st.table(df)
            else:
                st.warning("⚠️ Aucun résultat trouvé.")

    # Menu Visualisation
    elif menu == "🖼️ Visualisation":
        st.title(f"🖼️ Visualisation des Features - {dataset_name}")
        
        class_list = get_class_list(dataset_name, dataset)
        default_classes = class_list[:2] if len(class_list) > 1 else [class_list[0]]
        
        selected_classes = st.multiselect("🎯 Choisissez les classes :", class_list, default=default_classes)
        selected_features = st.text_input("🔢 Entrez les indices des features (ex: 469,35,89) :", "469,35,89")

        if st.button("📸 Afficher les visualisations"):
            try:
                features_list = [int(f.strip()) for f in selected_features.split(",")]
            except ValueError:
                st.error("❌ Format invalide ! Entrez des entiers séparés par des virgules.")
                st.stop()

            with st.spinner("🖼️ Génération des visualisations..."):
                torch.cuda.synchronize()
                for cls in selected_classes:
                    st.subheader(f"🔍 Classe : {cls}")
                    
                    if dataset_name == "ImageNet":
                        cls_idx = dataset.class_to_idx[cls]
                    else:
                        cls_idx = CIFAR10_CLASSES.index(cls)

                    for feature_id in features_list:
                        try:
                            ref_c = fv.get_stats_reference(
                                feature_id, "features.40", targets=cls_idx, mode="relevance",
                                r_range=(0, 6), rf=True, composite=composite, plot_fn=vis_opaque_img
                            )

                            if ref_c:
                                for img_list in ref_c.values():
                                    cols = st.columns(4)
                                    for idx, img in enumerate(img_list):
                                        if isinstance(img, torch.Tensor):
                                            img = T.ToPILImage()(img.cpu())
                                        if isinstance(img, Image.Image):
                                            col_idx = idx % 4
                                            cols[col_idx].image(img, caption=f"Feature {feature_id} - Img {idx+1}", use_container_width=True)
                                    st.write("---")
                            else:
                                st.warning(f"⚠️ Aucune image pour la feature {feature_id} et la classe {cls}")

                        except Exception as e:
                            st.error(f"⚠️ Erreur pour la feature {feature_id} de {cls} : {e}")

                torch.cuda.synchronize()
                st.success("✅ Visualisation terminée.")

if __name__ == "__main__":
    main()