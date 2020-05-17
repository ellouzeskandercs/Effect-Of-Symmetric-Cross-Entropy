import tensorflow as tf
import numpy as np


''' ARGS: metrics (metrics object), history (histroy object), dataset_type (str), loss_type (str), noise_type (str), noise_rate (float) '''
def save_metrics(metrics, history, dataset_type, loss_type, noise_type, noise_rate):
	curr_test_filename = str(dataset_type) + '_' + str(loss_type) + '_' + str(noise_type) + '_' + str(noise_rate) + '.txt'

	confidence = np.asarray(metrics.confidence)
	class_accuracies = np.array(metrics.train_acc_class).transpose()
	prediction_distribution = metrics.Pred
	prediction_distribution_correct = metrics.CorrPred
	overall_val_accuracy = metrics.acc

	np.savetxt('./Confidence_' + curr_test_filename, confidence)
	np.savetxt('./AccuracyPerClass_' + curr_test_filename, class_accuracies)
	np.savetxt('./PredictionDistr_' + curr_test_filename, prediction_distribution)
	np.savetxt('./PredictionDistrCorrect_' + curr_test_filename, prediction_distribution_correct)
	np.savetxt('./ValidationAccuracy_' + curr_test_filename, overall_val_accuracy)
