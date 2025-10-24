import pickle

if __name__ == "__main__":
    path1 = "datasets/neural/nsd/fmri_responses.pkl"
    path2 = "datasets/neural/nsd_streams/fmri_responses.pkl"

    # Load both pickle files
    with open(path1, "rb") as f:
        data1 = pickle.load(f)
    with open(path2, "rb") as f:
        data2 = pickle.load(f)

    print("Data1 keys:", data1.keys())
    print("Data2 keys:", data2.keys())
    print("Data1 V1 shape:", data1["V1"][0].shape)
    print("Data2 early visual stream shape:", data2["early visual stream"][0].shape)

    # Combine both dictionaries
    combined_data = {**data1, **data2}

    print("\nCombined keys:", combined_data.keys())

    # Save combined data to NSD folder
    output_path = "datasets/neural/nsd/fmri_responses_combined.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"\nCombined pickle file saved to: {output_path}")
