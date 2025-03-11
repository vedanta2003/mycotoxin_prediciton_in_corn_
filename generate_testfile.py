import pandas as pd
import numpy as np

# Load the original dataset
data = pd.read_csv("TASK-ML-INTERN.csv")

# Number of test files to generate
num_files = 3

# Number of samples per test file
samples_per_file = np.random.randint(10, 13)  # Random between 10 to 12

for i in range(1, num_files + 1):
    # Sample data randomly
    sampled_data = data.sample(n=samples_per_file, random_state=i)


    # Remove target column 
    test_data = sampled_data.iloc[:, :-1]  

    # Save to CSV
    file_name = f"test_samples_{i}.csv"
    test_data.to_csv(file_name, index=False)
    print(f"✅ Test file '{file_name}' created with {samples_per_file} samples.")

print("🎯 Test files generated successfully!")
