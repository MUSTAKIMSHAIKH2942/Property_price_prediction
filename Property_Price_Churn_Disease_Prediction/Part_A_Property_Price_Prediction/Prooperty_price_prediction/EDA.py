import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create a directory for saving the images if it doesn't exist
if not os.path.exists('eda_images'):
    os.makedirs('eda_images')

def eda(data):
    # Plotting the distribution of 'median_house_value'
    plt.figure()
    sns.histplot(data['median_house_value'], kde=True)
    plt.title("Distribution of Median House Value")
    plt.savefig('eda_images/median_house_value_distribution.png')  # Save the plot
    plt.close()
    
    # Plotting the relationship between 'median_income' and 'median_house_value'
    plt.figure()
    sns.scatterplot(x='median_income', y='median_house_value', data=data)
    plt.title("Median Income vs. House Value")
    plt.savefig('eda_images/median_income_vs_house_value.png')  # Save the plot
    plt.close()
    
    # Plotting the correlation matrix
    plt.figure()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig('eda_images/correlation_matrix.png')  # Save the plot
    plt.close()
