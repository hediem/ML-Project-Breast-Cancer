# ML-Project-Breast-Cancer

```markdown
# Breast Cancer Classification with Machine Learning Models  

This project explores various machine learning algorithms to classify breast cancer as benign or malignant using decision tree-based models and ensemble methods. The study evaluates models on accuracy, precision, recall, and F1-score, providing detailed insights into their performance and decision-making processes.  

## Features  
- **Algorithms Used:**  
  - CART (Classification and Regression Trees)  
  - C4.5  
  - AdaBoost  
  - XGBoost  
  - Random Forest  
  - LightGBM  
  - ExtraTrees  
  - Gradient Boosting  

- **Performance Metrics:**  
  - Confusion matrix  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  

- **Visualization:**  
  - Decision trees for selected models  
  - Comparative analysis of metrics  

## Dataset  
The project uses a publicly available dataset for breast cancer classification. It includes features extracted from cell nuclei images. Each instance is labeled as either benign (0) or malignant (1).  

## Project Structure  
```plaintext
├── dataset/                # Dataset files
├── src/                 # Source code for preprocessing and training
├── results/             # Generated reports and visualizations & Performance metrics and confusion matrices
├── README.md            # Project documentation

```

## Algorithms Overview  
1. **CART:** Uses binary splits to build interpretable decision trees.  
2. **C4.5:** An extension of ID3 with improvements like handling continuous data.  
3. **AdaBoost:** Combines weak learners iteratively to minimize classification error.  
4. **XGBoost:** Optimized gradient boosting with regularization.  
5. **Random Forest:** A robust ensemble of decision trees trained with random samples.  
6. **LightGBM:** A gradient-boosting framework optimized for speed and efficiency.  
7. **ExtraTrees:** Similar to Random Forest but uses more randomness in split selection.  
8. **Gradient Boosting:** Sequentially minimizes a loss function using weak learners.  

## Results Summary  
| Model              | Accuracy  | Number of Rules |
|---------------------|-----------|-----------------|
| CART               | 0.95238   | 67              |
| C4.5               | 0.95714   | 67              |
| AdaBoost           | 0.96190   | 4               |
| XGBoost1           | 0.96190   | 130             |
| Random Forest      | 0.97143   | 79              |
| LightGBM           | 0.95238   | N/A             |
| ExtraTrees         | 0.96667   | 130             |
| Gradient Boosting  | 0.96190   | N/A             |

## How to Use  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   cd breast-cancer-classification
   ```

2. Install the dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:  
   ```bash
   python src/main.py
   ```

## Conclusion  
This project highlights the strengths and limitations of tree-based algorithms for breast cancer classification. It provides valuable insights into model interpretability, efficiency, and accuracy.  


## Acknowledgments  
- [Dataset Source](https://www.kaggle.com/datasets/saurabhbadole/breast-cancer-wisconsin-state/data)  
- Inspired by various research and educational resources on decision tree algorithms.
``` 

Let me know if you’d like to customize further!
