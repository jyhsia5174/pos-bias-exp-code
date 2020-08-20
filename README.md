# pos-bias-exp-code

## Install dependencies

- mkl (https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)
- cuda 9.0 or higher
- anaconda python 3.7 version

## Create conda env for dl exps

```shell
conda env create -f pos.yml
conda activate py3.7
git clone https://github.com/johncreed/pos-bias-exp-code.git
git submodule init; git submodule update;
```

## Experiments

### Create data (should be do first!)

#### 1. Proprocess data

##### KKBOX

Put KKBOX's members.csv, songs.csv and train.csv to "data/data_preprocessing/kkbox-100/kkbox_csv_to_ocffm/".
```
cd data/data_preprocessing/kkbox-100/
./preprocess_kkbox.sh
```

##### Outbrain

Put Outbrain's clicks_train.csv, cv_events.csv, promote_content.csv and document_meta.csv to "data/data_preprocessing/ob.300/ob_csv_to_ocffm".
```
cd data/data_preprocessing/ob.300
./preprocess_ob.sh
```

#### 2. Final data preperation

#####  Create data with self validation set

```shell
cd data;
./gen_data_selfva.sh {raw_data_path} {pos_bias}
```

#####  Create data with rand validation set

```shell
cd data;
./gen_data_rndva.sh {raw_data_path} {pos_bias}
```

#### Some notes

- **{raw_data_path}** should have similar files following:
  - det.ffm.pos.0.5.bias  
  - item.ffm  
  - random.ffm.pos.0.5.bias  
  - truth.ffm
  - det.svm.pos.0.5.bias  
  - item.svm  
  - random.svm.pos.0.5.bias  
  - truth.svm
- **{pos_bias}** is like "0.5", "0.7".

### Run exps for $P(c=1 \mid i,j,k)$

#### For exps

```shell
cd run_dl_exp;
./exp_ffm.sh {data_path} {pos_bias} {gpu_idx} {mode}
```

- **{pos_bias}** is like "0.5", "0.7".
- **{gpu_idx}** should be one of GPU idxes of your current workstation, like "0" or "1";
- **{mode}** should be one of "logloss" and "auc".

#### Take results

```shell
cd run_dl_exp;
python get_record.py {exp_dir_for_specific_data}/{data_part} {mode}
```

### Run exps for hyffm

#### For exps

```shell
cd run_fm_exp;
./exp.sh {data_path} {mode} {pos_bias}
```

- **{pos_bias}** is like "0.5", "0.7".
- **{mode}** should be one of "logloss" and "auc".

#### Take results

- results of 'der.0.01,der.0.1,der.comb.0.01,der.comb.0.1,derive.det,derive.random'

    ```shell
    cd run_fm_exp;
    python get_record.py {exp_dir_for_specific_data}/{data_part} {mode}
    ```

- results of 'der.comb.0.01.imp,der.comb.0.1.imp'

    ```shell
    cd run_fm_exp;
    ./record-imp.sh {exp_dir_for_specific_data}/{data_part} {mode}
    ```

