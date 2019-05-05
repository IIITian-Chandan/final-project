import os
import sys
import glob
from clint.textui import colored


def add_paths(job_dir, model_id, env):
    model_id = model_id + "_1024"
    existing_weight = "None"

    os.system('sudo chmod go+rw ' + job_dir)
    os.system('sudo chmod +x ' + job_dir)
    os.system('sudo chmod +777 ' + job_dir)
    os.system('sudo chmod go+rw ' + job_dir + '/')
    os.system('sudo chmod +x ' + job_dir + '/')
    os.system('sudo chmod +777 ' + job_dir + '/')

    if not os.path.exists(job_dir):
        os.makedirs(job_dir, mode=0o600)

    if not os.path.exists(os.path.join(job_dir, model_id)):
        os.makedirs(os.path.join(job_dir, model_id), mode=0o600)
    job_dir = os.path.join(job_dir, model_id)

    os.system('sudo chmod go+rw ' + job_dir)
    os.system('sudo chmod +x ' + job_dir)
    os.system('sudo chmod +777 ' + job_dir)
    os.system('sudo chmod go+rw ' + job_dir + '/')
    os.system('sudo chmod +x ' + job_dir + '/')
    os.system('sudo chmod +777 ' + job_dir + '/')

    if not os.path.exists(os.path.join(job_dir, env)):
        os.makedirs(os.path.join(job_dir, env), mode=0o600)
    job_dir = os.path.join(job_dir, env)
    os.system('sudo chmod go+rw ' + job_dir)
    os.system('sudo chmod +x ' + job_dir)
    os.system('sudo chmod +777 ' + job_dir)
    os.system('sudo chmod go+rw ' + job_dir + '/')
    os.system('sudo chmod +x ' + job_dir + '/')
    os.system('sudo chmod +777 ' + job_dir + '/')

    if not os.path.exists(os.path.join(job_dir, "output")):
        os.makedirs(os.path.join(job_dir, "output"), mode=0o600)

    if not os.path.exists(os.path.join(job_dir, "weights")):
        os.makedirs(os.path.join(job_dir, "weights"), mode=0o600)
    else:
        existing_weights = glob.glob(os.path.join(job_dir, "weights") + "/*.h5")
        existing_weights = [i.split('/')[-1] for i in existing_weights]

        existing_weights_dict = {}
        for i, j in enumerate(existing_weights):
            existing_weights_dict[str(i + 1)] = j
        existing_weights_dict[str(len(existing_weights) + 1)] = "None"
        print(colored.green("Already you have " + str(len(existing_weights)) + " weights for this model."))
        for i in existing_weights_dict.keys():
            print(i + " : " + existing_weights_dict[i])

        print(colored.green("=========================="))
        print(colored.green("please choose an option  ="))
        print(colored.green("=========================="))

        option = input()
        existing_weight = existing_weights_dict[option]

    if not os.path.exists(os.path.join(job_dir, "logs")):
        os.makedirs(os.path.join(job_dir, "logs"), mode=0o600)

    os.system('sudo chmod go+rw ' + job_dir + '/output')
    os.system('sudo chmod +x ' + job_dir + '/output')
    os.system('sudo chmod +777 ' + job_dir + '/output')

    os.system('sudo chmod go+rw ' + job_dir + '/weights')
    os.system('sudo chmod +x ' + job_dir + '/weights')
    os.system('sudo chmod +777 ' + job_dir + '/weights')

    os.system('sudo chmod go+rw ' + job_dir + '/logs')
    os.system('sudo chmod +x ' + job_dir + '/logs')
    os.system('sudo chmod +777 ' + job_dir + '/logs')

    return job_dir, existing_weight
