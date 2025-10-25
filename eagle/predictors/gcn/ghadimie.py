
#                 handler.add_scalar('system/energy_consumption', metrics_collector.energy_consumption[-1], epoch_no)



#     # End of training

#     total_training_time = time.time() - total_training_start_time

    

#     if tensorboard:

#         handler.close()

    

#     if save:

#         torch.save(best_predictor_weight, outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt'))



#     # Save final results

#     cfg = {

#         'epochs': epochs,

#         'learning_rate': learning_rate,

#         'weight_decay': weight_decay,

#         'batch_size': batch_size

#     }

    

#     results = metrics_collector.get_final_results(cfg, model_name, total_training_time)

#     results_filepath = metrics_collector.save_results(results, outdir, model_name, exp_name)

    

#     # Create charts

#     charts_dir = metrics_collector.create_charts(results, outdir, model_name, exp_name)

    

#     print("Training finished!")

#     print(f"Results saved to: {results_filepath}")

#     if charts_dir:

#         print(f"Charts saved to: {charts_dir}")

    

#     # Print summary of final results

#     print("\n=== TRAINING SUMMARY ===")

#     print(f"Total training time: {total_training_time:.2f}s")

#     print(f"Best validation epoch: {results['best_epoch']}")

#     print(f"Best validation loss: {results['best_val_loss']:.6f}")

#     print(f"Final validation MAE: {results['mae']:.6f}")

#     print(f"Final validation R²: {results['r2']:.4f}")

#     print(f"Accuracy within 1%: {results['accuracy_1%']:.4f}")

#     print(f"Accuracy within 5%: {results['accuracy_5%']:.4f}")

#     print(f"Accuracy within 10%: {results['accuracy_10%']:.4f}")

#     print(f"Accuracy within 20%: {results['accuracy_20%']:.4f}")

#     if PSUTIL_AVAILABLE:

#         print(f"Average memory usage: {results['memory_usage_mb']:.2f} MB")

#         print(f"Average CPU usage: {results['avg_cpu_percent']:.2f}%")

#     if PYRAPL_AVAILABLE:

#         print(f"Total energy consumption: {results['total_energy_joules']:.2f} J")



#     predictor.load_state_dict(best_predictor_weight)

#     return predictor, results_filepath





# # The rest of the file remains exactly the same (predict function and main block)

# # [Previous predict function and main block code remains unchanged...]





# # The rest of the file remains the same (predict function and main block)

# # [Previous predict function and main block code remains unchanged...]



# def predict(testing_data,

#         outdir,

#         device_name,

#         model_name,

#         metric,

#         predictor_name,

#         predictor,

#         log=False,

#         exp_name=None,

#         load=False,

#         iteration=None,

#         explored_models=None,

#         valid_pts=None,

#         use_fast=True,

#         augments=None):

#     model_module = importlib.import_module('.' + model_name, 'eagle.models')



#     if load or log:

#         outdir = pathlib.Path(outdir) / model_name / metric / device_name / predictor_name

#         if log:

#             outdir.mkdir(parents=True, exist_ok=True)



#     if load and predictor_name != 'random':

#         predictor.load_state_dict(torch.load(outdir / ('predictor.pt' if exp_name is None else f'predictor_{exp_name}.pt')))

#         print('Predictor imported.')



#     criterion = torch.nn.L1Loss()



#     test_corrects = [0, 0, 0, 0]

#     leeways = [0.01, 0.05, 0.1, 0.2]



#     log_file = None

#     if log:

#         log_filename = 'log.txt' if exp_name is None else f'log_{exp_name}.txt'

#         if iteration is not None:

#             log_filename = f'iter{iteration}_' + log_filename



#         log_file = outdir / log_filename

#         sep = False

#         if log_file.exists():

#             sep = True

#         log_file = log_file.open('a')

#         if sep:

#             log_file.write('===\n')



#     if predictor_name == 'random':

#         print('Producing random ordering of the dataset...')

#         predicted = []

#         perm = np.random.permutation(len(testing_data))

#         for idx, (point, gt_value) in enumerate(testing_data):

#             predicted_value = perm[idx]

#             predicted.append(predicted_value)

#             if log:

#                 log_file.write(f'{gt_value} {predicted_value} {point}\n')



#     elif not predictor.binary_classifier:

#         predicted = []

#         test_loss = 0

#         for g, latency in testing_data:

#             corrects, loss, values = _test(model_module, predictor, g, latency, leeways, criterion, log_file, augments)

#             for i, c in enumerate(corrects):

#                 test_corrects[i] += c



#             test_loss += loss

#             predicted.append(values[1])



#         current_accuracies = [test_correct / len(testing_data) for test_correct in test_corrects]

#         avg_loss = test_loss / len(testing_data)



#         print(f'Top +-{leeways} Accuracy of test set: {current_accuracies}')

#         print(f'Average loss of test set: {avg_loss}')

#     else:

#         torch.set_grad_enabled(False)

#         predictor.eval()



#         if use_fast:

#             print(f'Precomputing embeddings for {len(testing_data)} graphs')

#             precomputed = infer.precompute_embeddings(model_module, predictor, testing_data, 1024, augments=augments)

#             print('Done')



#         total = 0

#         correct = 0

#         skipped = 0

#         def predictor_compare(v1, v2):

#             nonlocal total

#             nonlocal correct

#             nonlocal skipped

#             total += 1

#             if valid_pts is not None and v1[0] not in valid_pts:

#                 skipped += 1

#                 return -1

#             if valid_pts is not None and v2[0] not in valid_pts:

#                 skipped += 1

#                 return 1

#             latencies = [v1[1], v2[1]]

#             if use_fast:

#                 result = infer.precomputed_forward(predictor, [v1[2], v2[2]], precomputed)

#             else:

#                 gs = [v1[0], v2[0]]

#                 adjacency, features, _, aug = infer.prepare_tensors([gs], None, model_module, predictor.binary_classifier, False, augments=augments)

#                 if augments is not None:

#                     result = predictor(adjacency, features, aug)

#                 else:

#                     result = predictor(adjacency, features)

#             if predictor.binary_classifier == 'oneway' or predictor.binary_classifier == 'oneway-hard':

#                 v1_better = result[0][0].cpu().item() - 0.5

#                 if latencies[0] > latencies[1]:

#                     if v1_better > 0:

#                         correct += 1

#                 elif v1_better < 0:

#                     correct += 1



#                 return v1_better

#             else:

#                 rv1, rv2 = result[0][0].cpu().item(), result[0][1].cpu().item()

#                 if latencies[0] > latencies[1]:

#                     if rv1 > rv2:

#                         correct += 1

#                 elif rv1 < rv2:

#                     correct += 1



#                 # we want higher number to appear later (have higher "score"), so (v1 - v2) should get us the correct order

#                 return rv1 - rv2



#         if use_fast:

#             predictor.cpu()

#             test_data_with_indices = [(*v, idx) for idx, v in enumerate(testing_data)]

#             sorted_values = sorted(test_data_with_indices, key=functools.cmp_to_key(predictor_compare))

#             sorted_values = { pt: (gt,idx) for idx,(pt,gt,_) in enumerate(sorted_values) }

#             # predictor.cuda()

#         else:

#             sorted_values = sorted(testing_data, key=functools.cmp_to_key(predictor_compare))

#             sorted_values = { pt: (gt,idx) for idx,(pt,gt) in enumerate(sorted_values) }



#         predicted = []

#         for p, v in testing_data:

#             r = sorted_values[p][1]

#             predicted.append(r)

#             if log:

#                 log_file.write(f'{v} {r} {p}\n')



#         predictor.train()

#         torch.set_grad_enabled(True)



#     if log:

#         log_file.write('---\n')

#         explored_models = explored_models or []

#         for p,v in explored_models:

#             log_file.write(f'{p}\n')

#         log_file.write('---\n')

#         if predictor_name == 'random':

#             pass

#         elif not predictor.binary_classifier:

#             log_file.write(f'{avg_loss}\n{current_accuracies}\n')

#         else:

#             log_file.write(f'{correct}/{total} predictions correct\n')

#             log_file.write(f'{skipped}/{total} predictions skipped\n')

#         log_file.close()



#     return predicted





# if __name__ == '__main__':

#     # [Previous main block code remains unchanged...]

#     # (The main block should remain the same as in your original code)