import tensorflow as tf
import os
import json
from variant import VARIANT, get_env_from_name,  get_train, get_eval
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# tf.reset_default_graph()

if __name__ == '__main__':
    root_dir = VARIANT['log_path']
    os.makedirs(os.path.dirname(root_dir + "/config_out.json"), exist_ok=True)
    with open(root_dir + "/config_out.json", 'w') as fp:
        json.dump(VARIANT, fp)
    if VARIANT['train']:
        for i in range(VARIANT['start_of_trial'], VARIANT['start_of_trial']+VARIANT['num_of_trials']):
            VARIANT['log_path'] = root_dir +'/'+ str(i)
            print('logging to ' + VARIANT['log_path'])
            # print(VARIANT)
            train = get_train(VARIANT['algorithm_name'])
            print(train)
            train(VARIANT)

            tf.reset_default_graph()
    else:
        eval = get_eval(VARIANT['algorithm_name'])
        eval(VARIANT)

