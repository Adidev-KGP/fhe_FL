# Federated Learning with Fully Homomorphic Encryption (FHE)

## Training Logs for FHE MODEL

#### Epoch 1/50, Loss: 0.3515, Accuracy: 81.46%
#### Epoch 2/50, Loss: 0.4625, Accuracy: 83.66%
#### Epoch 3/50, Loss: 0.2105, Accuracy: 84.11%
#### Epoch 4/50, Loss: 0.3722, Accuracy: 84.29%
#### Epoch 5/50, Loss: 0.3786, Accuracy: 83.82%
#### Epoch 6/50, Loss: 0.5878, Accuracy: 83.81%
#### Epoch 7/50, Loss: 0.3872, Accuracy: 83.90%
#### Epoch 8/50, Loss: 0.2698, Accuracy: 84.02%
#### Epoch 9/50, Loss: 0.3323, Accuracy: 84.18%
#### Epoch 10/50, Loss: 0.3185, Accuracy: 84.39%
#### Epoch 11/50, Loss: 0.4512, Accuracy: 84.34%
#### Epoch 12/50, Loss: 0.4378, Accuracy: 84.51%
#### Epoch 13/50, Loss: 0.4253, Accuracy: 84.59%
#### Epoch 14/50, Loss: 0.3197, Accuracy: 84.60%
#### Epoch 15/50, Loss: 0.3211, Accuracy: 84.78%
#### Epoch 16/50, Loss: 0.3277, Accuracy: 84.74%
#### Epoch 17/50, Loss: 0.3913, Accuracy: 84.77%
#### Epoch 18/50, Loss: 0.3947, Accuracy: 84.90%
#### Epoch 19/50, Loss: 0.4442, Accuracy: 84.89%
#### Epoch 20/50, Loss: 0.4221, Accuracy: 84.75%
#### Epoch 21/50, Loss: 0.2066, Accuracy: 84.79%
#### Epoch 22/50, Loss: 0.4528, Accuracy: 84.98%
#### Epoch 23/50, Loss: 0.4212, Accuracy: 84.99%
#### Epoch 24/50, Loss: 0.4785, Accuracy: 84.98%
#### Epoch 25/50, Loss: 0.3412, Accuracy: 84.89%
#### Epoch 26/50, Loss: 0.2986, Accuracy: 84.96%
#### Epoch 27/50, Loss: 0.2944, Accuracy: 84.89%
#### Epoch 28/50, Loss: 0.3497, Accuracy: 84.95%
#### Epoch 29/50, Loss: 0.3284, Accuracy: 85.00%
#### Epoch 30/50, Loss: 0.3374, Accuracy: 85.21%
#### Epoch 31/50, Loss: 0.4219, Accuracy: 85.17%
#### Epoch 32/50, Loss: 0.3671, Accuracy: 84.80%
#### Epoch 33/50, Loss: 0.4718, Accuracy: 84.82%
#### Epoch 34/50, Loss: 0.3859, Accuracy: 84.85%
#### Epoch 35/50, Loss: 0.4896, Accuracy: 84.90%
#### Epoch 36/50, Loss: 0.3373, Accuracy: 85.08%
#### Epoch 37/50, Loss: 0.4716, Accuracy: 85.04%
#### Epoch 38/50, Loss: 0.3937, Accuracy: 85.03%
#### Epoch 39/50, Loss: 0.2216, Accuracy: 85.14%
#### Epoch 40/50, Loss: 0.1792, Accuracy: 85.14%
#### Epoch 41/50, Loss: 0.2846, Accuracy: 85.22%
#### Epoch 42/50, Loss: 0.4148, Accuracy: 85.24%
#### Epoch 43/50, Loss: 0.4764, Accuracy: 85.23%
#### Epoch 44/50, Loss: 0.2478, Accuracy: 85.14%
#### Epoch 45/50, Loss: 0.4340, Accuracy: 85.38%
#### Epoch 46/50, Loss: 0.2575, Accuracy: 85.17%
#### Epoch 47/50, Loss: 0.2568, Accuracy: 85.15%
#### Epoch 48/50, Loss: 0.4314, Accuracy: 85.22%
#### Epoch 49/50, Loss: 0.2224, Accuracy: 85.19%
#### Epoch 50/50, Loss: 0.3818, Accuracy: 85.20%

## Training Time

#### Training time: 179.7990 seconds

## User Input from Terminal

#### Please enter the following details:
#### person_age: 30
#### person_income: 200000
#### person_emp_length: 10
#### loan_amnt: 20000000
#### loan_int_rate: 8
#### loan_percent_income: 10
#### cb_person_cred_hist_length: 10

## Encrypted prediction:

#### Encrypted prediction: [[0.03041826]]



## Time Taken for encrypted inference:

####  Inference time: 20.3751 seconds


##############

## Training Logs for Classical Model in FL Setup:

### server side logs
```
python3 server.py 

-------------------------------

INFO :      Starting Flower server, config: num_rounds=1, no round_timeout
INFO :      Flower ECE: gRPC server running (1 rounds), SSL is disabled
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
INFO :      Received initial parameters from one random client
INFO :      Evaluating initial global parameters
INFO :      
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 2 clients (out of 2)
INFO :      aggregate_fit: received 2 results and 0 failures
WARNING :   No fit_metrics_aggregation_fn provided
INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 1 round(s) in 69.46s
INFO :          History (loss, distributed):
INFO :                  round 1: 0.3429020047187805
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 0.8584148287773132)]}
INFO :      

```

### client-1 side logs
```
python3 client.py --partition-id 0

------------------------------------------------
Please enter the following details:
person_age: 30
person_income: 5454
person_emp_length: 10
loan_amnt: 452353
loan_int_rate: 10
loan_percent_income: 8
cb_person_cred_hist_length: 10

------------------------------------------------

1/1 [==============================] - 0s 126ms/step
Initial prediction (before federated learning): 0.7143
INFO :      
INFO :      Received: get_parameters message 052acffc-59b4-425c-9a94-cb1a386a8f1e
INFO :      Sent reply
INFO :      
INFO :      Received: train message 3612b184-9e14-4369-a7dd-edc46174be6f
Epoch 1/50
716/716 [==============================] - 2s 1ms/step - loss: 0.4002 - accuracy: 0.8310
Epoch 2/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3668 - accuracy: 0.8442
Epoch 3/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3615 - accuracy: 0.8463
Epoch 4/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3582 - accuracy: 0.8493
Epoch 5/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3564 - accuracy: 0.8498
Epoch 6/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3551 - accuracy: 0.8513
Epoch 7/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3530 - accuracy: 0.8508
Epoch 8/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3514 - accuracy: 0.8535
Epoch 9/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3502 - accuracy: 0.8532
Epoch 10/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3487 - accuracy: 0.8540
Epoch 11/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3481 - accuracy: 0.8542
Epoch 12/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3469 - accuracy: 0.8550
Epoch 13/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3460 - accuracy: 0.8565
Epoch 14/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3450 - accuracy: 0.8574
Epoch 15/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3443 - accuracy: 0.8570
Epoch 16/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3438 - accuracy: 0.8567
Epoch 17/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3433 - accuracy: 0.8590
Epoch 18/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3420 - accuracy: 0.8591
Epoch 19/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3419 - accuracy: 0.8598
Epoch 20/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3407 - accuracy: 0.8593
Epoch 21/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3394 - accuracy: 0.8594
Epoch 22/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3389 - accuracy: 0.8602
Epoch 23/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3383 - accuracy: 0.8613
Epoch 24/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3376 - accuracy: 0.8632
Epoch 25/50
716/716 [==============================] - 1s 2ms/step - loss: 0.3369 - accuracy: 0.8627
Epoch 26/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3367 - accuracy: 0.8635
Epoch 27/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3355 - accuracy: 0.8626
Epoch 28/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3350 - accuracy: 0.8615
Epoch 29/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3344 - accuracy: 0.8635
Epoch 30/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3339 - accuracy: 0.8625
Epoch 31/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3329 - accuracy: 0.8645
Epoch 32/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3333 - accuracy: 0.8648
Epoch 33/50
716/716 [==============================] - 1s 2ms/step - loss: 0.3322 - accuracy: 0.8641
Epoch 34/50
716/716 [==============================] - 1s 2ms/step - loss: 0.3313 - accuracy: 0.8659
Epoch 35/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3305 - accuracy: 0.8658
Epoch 36/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3306 - accuracy: 0.8648
Epoch 37/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3292 - accuracy: 0.8662
Epoch 38/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3290 - accuracy: 0.8674
Epoch 39/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3288 - accuracy: 0.8653
Epoch 40/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3275 - accuracy: 0.8668
Epoch 41/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3277 - accuracy: 0.8682
Epoch 42/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3274 - accuracy: 0.8672
Epoch 43/50
716/716 [==============================] - 1s 2ms/step - loss: 0.3266 - accuracy: 0.8673
Epoch 44/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3262 - accuracy: 0.8665
Epoch 45/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3261 - accuracy: 0.8677
Epoch 46/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3257 - accuracy: 0.8684
Epoch 47/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3245 - accuracy: 0.8688
Epoch 48/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3250 - accuracy: 0.8692
Epoch 49/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3236 - accuracy: 0.8689
Epoch 50/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3233 - accuracy: 0.8695
Training time: 50.8179 seconds
INFO :      Sent reply
INFO :      
INFO :      Received: evaluate message a2452cde-997b-4be6-ba1f-cbef7ad4d45a
179/179 [==============================] - 0s 1ms/step - loss: 0.3429 - accuracy: 0.8584
INFO :      Sent reply
INFO :      
INFO :      Received: reconnect message 8683b6cf-3d70-4a8f-8467-3d6416cac5b9
INFO :      Disconnect and shut down
1/1 [==============================] - 0s 17ms/step
Final prediction (after federated learning): 0.6504702171943707
Inference time: 0.0772 seconds

```

### client-2 side logs
```
python3 client.py --partition-id 1

-----------------------------------------------

Please enter the following details:
person_age: 35
person_income: 100000
person_emp_length: 16
loan_amnt: 400428855
loan_int_rate: 10
loan_percent_income: 11
cb_person_cred_hist_length: 16

------------------------------------------------

1/1 [==============================] - 0s 80ms/step
Initial prediction (before federated learning): 1.4863
INFO :      
INFO :      Received: train message 06f28c17-ba68-496c-8bcb-781294bb0953
Epoch 1/50
716/716 [==============================] - 2s 1ms/step - loss: 0.4018 - accuracy: 0.8309
Epoch 2/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3675 - accuracy: 0.8456
Epoch 3/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3608 - accuracy: 0.8482
Epoch 4/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3579 - accuracy: 0.8477
Epoch 5/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3570 - accuracy: 0.8504
Epoch 6/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3542 - accuracy: 0.8517
Epoch 7/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3530 - accuracy: 0.8520
Epoch 8/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3514 - accuracy: 0.8525
Epoch 9/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3502 - accuracy: 0.8539
Epoch 10/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3496 - accuracy: 0.8546
Epoch 11/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3492 - accuracy: 0.8540
Epoch 12/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3482 - accuracy: 0.8564
Epoch 13/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3470 - accuracy: 0.8560
Epoch 14/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3466 - accuracy: 0.8572
Epoch 15/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3459 - accuracy: 0.8567
Epoch 16/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3449 - accuracy: 0.8570
Epoch 17/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3444 - accuracy: 0.8581
Epoch 18/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3439 - accuracy: 0.8576
Epoch 19/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3436 - accuracy: 0.8589
Epoch 20/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3426 - accuracy: 0.8591
Epoch 21/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3419 - accuracy: 0.8590
Epoch 22/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3419 - accuracy: 0.8590
Epoch 23/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3410 - accuracy: 0.8595
Epoch 24/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3399 - accuracy: 0.8606
Epoch 25/50
716/716 [==============================] - 1s 2ms/step - loss: 0.3398 - accuracy: 0.8605
Epoch 26/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3389 - accuracy: 0.8615
Epoch 27/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3384 - accuracy: 0.8617
Epoch 28/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3371 - accuracy: 0.8631
Epoch 29/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3370 - accuracy: 0.8612
Epoch 30/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3368 - accuracy: 0.8622
Epoch 31/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3356 - accuracy: 0.8638
Epoch 32/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3346 - accuracy: 0.8629
Epoch 33/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3336 - accuracy: 0.8644
Epoch 34/50
716/716 [==============================] - 1s 2ms/step - loss: 0.3333 - accuracy: 0.8638
Epoch 35/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3327 - accuracy: 0.8661
Epoch 36/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3326 - accuracy: 0.8636
Epoch 37/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3320 - accuracy: 0.8651
Epoch 38/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3314 - accuracy: 0.8646
Epoch 39/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3302 - accuracy: 0.8660
Epoch 40/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3301 - accuracy: 0.8666
Epoch 41/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3293 - accuracy: 0.8649
Epoch 42/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3298 - accuracy: 0.8658
Epoch 43/50
716/716 [==============================] - 1s 2ms/step - loss: 0.3283 - accuracy: 0.8661
Epoch 44/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3267 - accuracy: 0.8661
Epoch 45/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3270 - accuracy: 0.8676
Epoch 46/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3269 - accuracy: 0.8677
Epoch 47/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3265 - accuracy: 0.8674
Epoch 48/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3256 - accuracy: 0.8681
Epoch 49/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3248 - accuracy: 0.8679
Epoch 50/50
716/716 [==============================] - 1s 1ms/step - loss: 0.3252 - accuracy: 0.8683
Training time: 50.6416 seconds
INFO :      Sent reply
INFO :      
INFO :      Received: evaluate message e2dab358-e6e5-4466-badc-42f2f1caaf77
179/179 [==============================] - 0s 1ms/step - loss: 0.3429 - accuracy: 0.8584
INFO :      Sent reply
INFO :      
INFO :      Received: reconnect message 139d79bf-6e33-4561-aefc-b2ab0d7e8082
INFO :      Disconnect and shut down
1/1 [==============================] - 0s 17ms/step
Final prediction (after federated learning): 0.29480993835526054
Inference time: 0.0772 seconds

```

### Client-1 benchmarks

#### Training time: 50.8179 seconds
#### Final prediction (after federated learning): 0.6504702171943707
#### Inference time: 0.0772 seconds

### Client-2 benchmarks

#### Training time: 50.6416 seconds
#### Final prediction (after federated learning): 0.29480993835526054
#### Inference time: 0.0772 seconds


## Deviation on FHE v/s Classical ML

### Input Details:

person_age: 30
person_income: 5454
person_emp_length: 10
loan_amnt: 452353
loan_int_rate: 10
loan_percent_income: 8
cb_person_cred_hist_length: 10

### Fhe prediction : 0.45704926
### Classical ML prediction : 0.6504702171943707

### Deviation : 29.735%