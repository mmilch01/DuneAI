###################################################################
## Import DL packages setting up GPU if used
import os
import tensorflow as tf
import keras.backend as K
from TheDuneAI import ContourPilot as cp 
import argparse


model_path=''

def init_env():
    global model_path
    script_dir=os.path.dirname(os.path.realpath(__file__))

    #If you have an available GPU and tensorflow-gpu >=1.15.0, CUDA >= 10.0.130, CuDNN installed you can try setting gpu=True
    GPU_compute = True
    if GPU_compute:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        os.environ['CUDA_VISIBLE_DEVICES']='0'          #Choose GPU device ID
        #Check available GPUs
        #print(K.tensorflow_backend._get_available_gpus())
        print(tf.config.list_physical_devices('GPU'))

        ###################################################################
        ##Initialize model inputs
        model_path=f'{script_dir}/model_files/'   #path to the model files

        #path_to_test_data = r'/workspace/admin/RIDER-LUNG-CT' #path to the data to be segmented (nrrds)
        #save_path ='/workspace/admin/RIDER-LUNG-CT/duneai_seg'

#subjects=["RIDER-1129164940","RIDER-1129164940","RIDER-1225316081","RIDER-1225316081"]
#sessions=["09-20-2006-1-NA-96508","09-20-2006-1-NA-96508","01-30-2007-NA-NA-56138","01-30-2007-NA-NA-56138"]
#labels=["TEST","RETEST","TEST","RETEST"]

#patient_dict={}
#for i in range(0,4):
#    patient_dict[i]=[ f"{path_to_test_data}/{subjects[i]}/{subjects[i]}_{labels[i]}_struct.nii" ]

#print(patient_dict)

#r'./produced segmentations' #path for the output files (nrrds)
#initialize the model

###################################################################
## Estimated segmentation time per patient:
##      with GPU(RTX2080TI): 2-3 sec
##      with CPU(Core i5-7200U ): 170 -180 sec
## Estimated processing time per patient depends on the multiple parameters such as: CPU/GPU usage, Hardware (HDD/SSD),
## and length of the CT scan (whole body scan CT/ thorax CT) 
## The estimated processing time per pat. range is: 25 sec - 280 sec.
###################################################################
##Starting the segmentation process

#model.segment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT Struct converter")
    parser.add_argument("infiles", type=str,help="Subject list, comma separated")
    parser.add_argument("inpath", type=str,help="Path output images directory")
    parser.add_argument("outpath", type=str,help="Path output images directory")
    #parser.add_argument("sessions", type=str,help="Session list, comma separated")
    #parser.add_argument("outfiles", type=str,help="Scan label list, comma separated")    
    init_env()
    args=parser.parse_args()
    path_to_data=args.inpath
    #subjs=args.subjects.split(',')
    #exps=args.sessions.split(',')
    infiles=args.infiles.split(',')
    #labels=args.labels.split(',')
    patient_dict={}
    for i in range(0,len(infiles)):
        patient_dict[i]=[ infiles[i] ]

    print(patient_dict)
    model = cp(model_path,path_to_data,args.outpath,verbosity=True,pat_dict=patient_dict)    #set verbosity =True to see what is going on
    model.segment()
    print('Segmentation batch finished.')
