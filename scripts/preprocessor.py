from src.preprocessor import E1_DAIC

def main():
    e1_daic = E1_DAIC('datasets/DAIC-WOZ/', 'datasets/E-DAIC-WOZ/', 'datasets/E1-DAIC-WOZ/')
    train, test, dev = e1_daic.get_dataset_splits()

    for dataset in [train, dev, test]:
        print(f"\nDataset: {dataset['Split'].values[0]}")
        class_percentages = dataset["PHQ_Binary"].value_counts(normalize=True) * 100
    
        print("\nPercentuali:")
        for class_val, percentage in class_percentages.items():
            print(f"  Classe {class_val}: {percentage:.2f}%")
        
        print(f"\nTotale campioni: {len(dataset)}")

        e1_daic.print_audio_duration_stats(dataset)

if __name__ == "__main__":
    main()