import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BASE_DIR, 'logs', 'evaluation')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'saved_model', 'best_model.h5')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'test')

os.makedirs(EVAL_DIR, exist_ok=True)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def preprocess_image(image, label):
    image = image / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    image = (image - mean) / std
    return image, label

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def calibrate_model(logits, y_true):
    def eval_temp(temp):
        scaled_logits = logits / temp
        probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
        return expected_calibration_error(y_true, probs)
        
    res = scipy.optimize.minimize_scalar(eval_temp, bounds=(0.1, 5.0), method='bounded')
    return res.x, res.fun

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Test directory not found: {TEST_DIR}")
        return

    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, label_mode='categorical', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    class_names = test_ds_raw.class_names
    test_ds = test_ds_raw.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Running evaluation...")
    y_true = np.concatenate([y for x, y in test_ds.as_numpy_iterator()])
    y_true_labels = np.argmax(y_true, axis=1)
    
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                                    outputs=model.layers[-2].output)
    try:
        features = intermediate_layer_model.predict(test_ds)
        logits = np.dot(features, model.layers[-1].get_weights()[0]) + model.layers[-1].get_weights()[1]
    except Exception:
        y_pred_prob = model.predict(test_ds)
        logits = np.log(y_pred_prob + 1e-15)
        
    y_pred_prob = model.predict(test_ds)
    y_pred_labels = np.argmax(y_pred_prob, axis=1)

    cr = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)
    with open(os.path.join(EVAL_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump(cr, f, indent=4)
    with open(os.path.join(EVAL_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_perc = cm / np.sum(cm, axis=1, keepdims=True)
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                annot[i, j] = f'{c}\n{p:.1%}'
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = f'{c}\n{p:.1%}'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'), dpi=300)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        cr[class_name]['auc'] = roc_auc
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, 'roc_curves.png'), dpi=300)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
        plt.plot(recall, precision, label=class_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, 'pr_curves.png'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, class_name in enumerate(class_names):
        correct = (y_true_labels == i) & (y_pred_labels == i)
        incorrect = (y_true_labels == i) & (y_pred_labels != i)
        
        conf_correct = y_pred_prob[correct, i]
        conf_incorrect = y_pred_prob[incorrect, i]
        
        axes[i].hist(conf_correct, bins=20, alpha=0.5, label='Correct')
        axes[i].hist(conf_incorrect, bins=20, alpha=0.5, label='Incorrect')
        axes[i].set_title(class_name)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'confidence_histograms.png'))

    incorrect_idxs = np.where(y_true_labels != y_pred_labels)[0]
    file_paths = test_ds_raw.file_paths
    
    worst_idxs = []
    for i in range(len(class_names)):
        class_incorrects = [idx for idx in incorrect_idxs if y_true_labels[idx] == i]
        if not class_incorrects:
            continue
        confs = [y_pred_prob[idx, y_pred_labels[idx]] for idx in class_incorrects]
        sorted_indices = np.argsort(confs)[:3]
        class_worst = [class_incorrects[j] for j in sorted_indices]
        worst_idxs.extend(class_worst)
        
    plt.figure(figsize=(15, 12))
    for k, idx in enumerate(worst_idxs[:12]):
        plt.subplot(4, 3, k+1)
        img = tf.keras.utils.load_img(file_paths[idx], target_size=(224, 224))
        plt.imshow(img)
        t_cls = class_names[y_true_labels[idx]]
        p_cls = class_names[y_pred_labels[idx]]
        conf = y_pred_prob[idx, y_pred_labels[idx]]
        plt.title(f"True: {t_cls}\nPred: {p_cls} ({conf:.2f})")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'misclassified_examples.png'))

    ece_before = expected_calibration_error(y_true_labels, y_pred_prob)
    opt_temp, ece_after = calibrate_model(logits, y_true_labels)
    with open(os.path.join(BASE_DIR, 'model', 'temperature.json'), 'w') as f:
        json.dump({'temperature': float(opt_temp)}, f)
        
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = np.max(y_pred_prob, axis=1)
    accuracies = y_pred_labels == y_true_labels
    
    bin_accs = []
    bin_confs = []
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bl) & (confidences <= bu)
        if np.any(in_bin):
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            
    plt.figure(figsize=(8,8))
    plt.plot(bin_confs, bin_accs, marker='o', label='Model')
    plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction Positives')
    plt.title('Calibration Plot (Reliability Diagram)')
    plt.text(0.1, 0.9, f'ECE: {ece_before:.4f}', bbox=dict(facecolor='white', alpha=0.5))
    plt.legend()
    plt.savefig(os.path.join(EVAL_DIR, 'calibration_plot.png'))

    acc = cr['accuracy']
    weighted_f1 = cr['weighted avg']['f1-score']
    macro_auc = np.mean([cr[c]['auc'] for c in class_names])

    print("══════════════════════════════════════")
    print("EVALUATION SUMMARY — SkinAI")
    print("══════════════════════════════════════")
    print(f"Test Accuracy:        {acc*100:.2f}%")
    print(f"Weighted F1-Score:    {weighted_f1:.3f}")
    print(f"Macro AUC-ROC:        {macro_auc:.3f}")
    print("──────────────────────────────────────")
    print("Per-Class Results:")
    print(f"{'Class':<20} {'Prec':<6} {'Rec':<6} {'F1':<6} {'AUC':<6}")
    for c in class_names:
        print(f"{c:<20} {cr[c]['precision']:.2f}   {cr[c]['recall']:.2f}   {cr[c]['f1-score']:.2f}   {cr[c]['auc']:.2f}")
    print("══════════════════════════════════════")
    print("All evaluation artifacts saved to: logs/evaluation/")
    
    print("\n┌─────────────────────────────────────────┐")
    print("│ CALIBRATION RESULTS                     │")
    print("├─────────────────────────────────────────┤")
    print(f"│ ECE Before Calibration: {ece_before:.4f}          │")
    print(f"│ ECE After Temp Scaling: {ece_after:.4f}          │")
    print(f"│ Optimal Temperature T:  {opt_temp:.2f}            │")
    print("└─────────────────────────────────────────┘")

if __name__ == '__main__':
    main()
