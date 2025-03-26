config = {
	'scale': 0.5,
	'cliplow' : -1000,
	'cliphigh' : 800,
	'batch_size' : 4, #torch.cuda.device_count()
	'num_epochs': 10,
	'num_workers': 8,
	'num_classes': 4,
	'val_ratio': 0.1,
	'save_every': 5,
    'updated_interval': 2,
    'frozen_iter': 2,
    'removing_ratio': 0.5,
}