import os
import numpy as np
import argparse

def calculate_average(embeddings):
    return np.mean(embeddings, axis=0)

def main(input_folder, output_file):
    embeddings = []
    
    for file in os.listdir(input_folder):
        if file.endswith(".npy"):
            filepath = os.path.join(input_folder, file)
            embedding = np.load(filepath)
            embeddings.append(embedding)

    embeddings = np.stack(embeddings)
    average_embedding = calculate_average(embeddings)
    np.save(output_file, average_embedding, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average embedding from a folder of .npy files.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing the .npy files.")
    parser.add_argument("output_file", type=str, help="Path to the output .npy file containing the average embedding.")
    args = parser.parse_args()

    main(args.input_folder, args.output_file)