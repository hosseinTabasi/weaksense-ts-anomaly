# WeakSense — Results

All numbers are aggregated from `experiments/results/*_raw.csv` (seeds 0,1,2; mean ± std).

## E1 Label-budget curve (SKAB test runs)

| setting      | method                                 |   auroc_mean |   auroc_std |   auprc_mean |   auprc_std |   f1@0.5_mean |   f1@0.5_std |
|:-------------|:---------------------------------------|-------------:|------------:|-------------:|------------:|--------------:|-------------:|
| budget=0.0   | LF label model -> encoder              |        0.845 |       0.008 |        0.747 |       0.019 |         0.635 |        0.013 |
| budget=0.0   | PU (nnPU)                              |        0.836 |       0.007 |        0.683 |       0.009 |         0.641 |        0.013 |
| budget=0.0   | WeakSense (LF+labels+curriculum)       |        0.802 |       0.051 |        0.661 |       0.058 |         0.639 |        0.008 |
| budget=0.005 | Supervised (budget only)               |        0.617 |       0.086 |        0.61  |       0.105 |         0.399 |        0.188 |
| budget=0.005 | PU (nnPU)                              |        0.847 |       0.005 |        0.69  |       0.011 |         0.644 |        0.014 |
| budget=0.005 | WeakSense (LF+labels+curriculum)       |        0.834 |       0.027 |        0.752 |       0.053 |         0.645 |        0.004 |
| budget=0.01  | Supervised (budget only)               |        0.738 |       0.027 |        0.764 |       0.018 |         0.68  |        0.018 |
| budget=0.01  | PU (nnPU)                              |        0.832 |       0.002 |        0.674 |       0.011 |         0.635 |        0.013 |
| budget=0.01  | WeakSense (LF+labels+curriculum)       |        0.882 |       0.004 |        0.828 |       0.012 |         0.658 |        0.003 |
| budget=0.05  | Supervised (budget only)               |        0.851 |       0.015 |        0.829 |       0.011 |         0.738 |        0.014 |
| budget=0.05  | PU (nnPU)                              |        0.872 |       0.022 |        0.746 |       0.038 |         0.664 |        0.017 |
| budget=0.05  | WeakSense (LF+labels+curriculum)       |        0.882 |       0.022 |        0.837 |       0.035 |         0.689 |        0.014 |
| budget=0.1   | Supervised (budget only)               |        0.809 |       0.015 |        0.793 |       0.02  |         0.741 |        0.014 |
| budget=0.1   | PU (nnPU)                              |        0.872 |       0.019 |        0.738 |       0.036 |         0.694 |        0.009 |
| budget=0.1   | WeakSense (LF+labels+curriculum)       |        0.892 |       0.006 |        0.842 |       0.004 |         0.764 |        0.015 |
| budget=1.0   | Fully supervised                       |        0.852 |       0.037 |        0.774 |       0.031 |         0.732 |        0.004 |
| unsupervised | IsolationForest                        |        0.854 |       0.012 |        0.777 |       0.017 |       nan     |      nan     |
| unsupervised | Dense autoencoder                      |        0.726 |       0.009 |        0.651 |       0.01  |       nan     |      nan     |
| unsupervised | LF majority vote (no training)         |        0.755 |       0     |        0.627 |       0     |         0.613 |        0     |
| unsupervised | LF Dawid-Skene posterior (no training) |        0.813 |       0     |        0.696 |       0     |         0.628 |        0     |

## E2 e2_lf_analysis

| lf                           |   coverage |   overlap |   conflict |   empirical_acc | polarity   |   learned_acc |
|:-----------------------------|-----------:|----------:|-----------:|----------------:|:-----------|--------------:|
| threshold_any_sensor>5sd     |      0.63  |     0.63  |      0.302 |           0.503 | 1          |         0.5   |
| all_sensors_within_2sd       |      0.001 |     0.001 |      0.001 |           1     | 0          |         0.694 |
| rate_of_change>3sd           |      0.979 |     0.976 |      0.549 |           0.348 | 1          |         0.5   |
| cross_sensor_inconsistency   |      0.825 |     0.825 |      0.426 |           0.44  | 0/1        |         0.572 |
| spectral_high_freq_share>0.6 |      0.923 |     0.923 |      0.529 |           0.358 | 1          |         0.5   |
| mean_shift                   |      0.496 |     0.496 |      0.281 |           0.715 | 0/1        |         0.99  |
| variance_burst               |      0.44  |     0.44  |      0.389 |           0.743 | 0/1        |         0.608 |
| threshold_sensor0            |      0.16  |     0.16  |      0.076 |           0.731 | 0/1        |         0.916 |
| threshold_sensor1            |      0.113 |     0.113 |      0.034 |           0.813 | 0/1        |         0.974 |
| threshold_sensor2            |      0.03  |     0.03  |      0.03  |           0.746 | 0          |         0.494 |
| threshold_sensor3            |      0.125 |     0.125 |      0.125 |           0.591 | 0/1        |         0.498 |
| threshold_sensor4            |      0.476 |     0.476 |      0.258 |           0.493 | 0/1        |         0.753 |
| threshold_sensor5            |      0.496 |     0.496 |      0.282 |           0.553 | 0/1        |         0.873 |
| threshold_sensor6            |      0.002 |     0.002 |      0.002 |           0.824 | 0          |         0.446 |
| threshold_sensor7            |      0.226 |     0.226 |      0.129 |           0.792 | 0/1        |         0.882 |

## E2 e2_label_model_quality

| label_model                   |   auroc |   auprc |   f1@0.5 |
|:------------------------------|--------:|--------:|---------:|
| Majority vote                 |   0.749 |   0.615 |    0.57  |
| Dawid-Skene EM                |   0.815 |   0.709 |    0.583 |
| LF coverage (any non-abstain) | nan     | nan     |    1     |

## E3 Noisy / adversarial LFs

| kind        |   n_bad_lfs |   mv_train_auroc_mean |   mv_train_auroc_std |   ds_train_auroc_mean |   ds_train_auroc_std |   ds_anchored_train_auroc_mean |   ds_anchored_train_auroc_std |   weaksense_test_auroc_mean |   weaksense_test_auroc_std |
|:------------|------------:|----------------------:|---------------------:|----------------------:|---------------------:|-------------------------------:|------------------------------:|----------------------------:|---------------------------:|
| random      |           0 |                 0.749 |                0     |                 0.815 |                    0 |                          0.819 |                         0.001 |                       0.882 |                      0.004 |
| random      |           2 |                 0.722 |                0.002 |                 0.814 |                    0 |                          0.818 |                         0.001 |                       0.881 |                      0.007 |
| random      |           4 |                 0.713 |                0.001 |                 0.814 |                    0 |                          0.818 |                         0.001 |                       0.879 |                      0.013 |
| random      |           8 |                 0.696 |                0.002 |                 0.814 |                    0 |                          0.818 |                         0.001 |                       0.88  |                      0.004 |
| adversarial |           0 |                 0.749 |                0     |                 0.815 |                    0 |                          0.819 |                         0.001 |                       0.882 |                      0.004 |
| adversarial |           2 |                 0.749 |                0     |                 0.815 |                    0 |                          0.819 |                         0.001 |                       0.881 |                      0.008 |
| adversarial |           4 |                 0.749 |                0     |                 0.815 |                    0 |                          0.819 |                         0.001 |                       0.882 |                      0.012 |
| adversarial |           8 |                 0.749 |                0     |                 0.815 |                    0 |                          0.819 |                         0.001 |                       0.884 |                      0.01  |

## E4 Transfer

| transfer                                    | method                           |   auroc_mean |   auroc_std |   auprc_mean |   auprc_std |   f1@0.5_mean |   f1@0.5_std |
|:--------------------------------------------|:---------------------------------|-------------:|------------:|-------------:|------------:|--------------:|-------------:|
| SKAB valve1 -> valve2+other                 | Supervised (budget only)         |        0.55  |       0.027 |        0.527 |       0.025 |         0.427 |        0.023 |
| SKAB valve1 -> valve2+other                 | LF label model -> encoder        |        0.767 |       0.019 |        0.561 |       0.02  |         0.648 |        0.017 |
| SKAB valve1 -> valve2+other                 | PU (nnPU)                        |        0.685 |       0.064 |        0.508 |       0.012 |         0.587 |        0.049 |
| SKAB valve1 -> valve2+other                 | WeakSense (LF+labels+curriculum) |        0.785 |       0.038 |        0.614 |       0.044 |         0.656 |        0.022 |
| SKAB valve1 -> valve2+other                 | Fully supervised (source)        |        0.8   |       0.025 |        0.659 |       0.032 |         0.548 |        0.025 |
| SKAB valve1 -> valve2+other                 | IsolationForest                  |        0.797 |       0.006 |        0.732 |       0.012 |       nan     |      nan     |
| TEP faults {1,2,4-8,10-12} -> {13,14,16-21} | Supervised (budget only)         |        0.727 |       0.018 |        0.939 |       0.006 |         0.922 |        0.002 |
| TEP faults {1,2,4-8,10-12} -> {13,14,16-21} | LF label model -> encoder        |        0.895 |       0.011 |        0.981 |       0.003 |         0.933 |        0.005 |
| TEP faults {1,2,4-8,10-12} -> {13,14,16-21} | PU (nnPU)                        |        0.738 |       0.042 |        0.947 |       0.01  |         0.854 |        0.052 |
| TEP faults {1,2,4-8,10-12} -> {13,14,16-21} | WeakSense (LF+labels+curriculum) |        0.867 |       0.028 |        0.974 |       0.007 |         0.911 |        0.021 |
| TEP faults {1,2,4-8,10-12} -> {13,14,16-21} | Fully supervised (source)        |        0.884 |       0.007 |        0.978 |       0.003 |         0.863 |        0.019 |
| TEP faults {1,2,4-8,10-12} -> {13,14,16-21} | IsolationForest                  |        0.886 |       0.003 |        0.98  |       0.001 |       nan     |      nan     |

## E5 Ablation (budget 1%)

| variant                              |   auroc_mean |   auroc_std |   auprc_mean |   auprc_std |   f1@0.5_mean |   f1@0.5_std |
|:-------------------------------------|-------------:|------------:|-------------:|------------:|--------------:|-------------:|
| Full WeakSense                       |        0.882 |       0.004 |        0.828 |       0.012 |         0.658 |        0.003 |
| No LFs (labels + self-training)      |        0.818 |       0.048 |        0.806 |       0.018 |         0.727 |        0.02  |
| No self-training                     |        0.868 |       0.006 |        0.809 |       0.016 |         0.637 |        0.017 |
| No curriculum (fixed threshold)      |        0.874 |       0.023 |        0.838 |       0.024 |         0.657 |        0.011 |
| Majority vote instead of Dawid-Skene |        0.847 |       0.006 |        0.768 |       0.025 |         0.602 |        0.017 |
| No true labels (budget 0)            |        0.802 |       0.051 |        0.661 |       0.058 |         0.639 |        0.008 |
