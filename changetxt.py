import os
import glob
from tqdm import tqdm
import soundfile as sf
# print('hello')
AUDIO_PATH='data/'
def change():

   
    for partition in sorted(glob.glob(AUDIO_PATH+'/*')):
            # print(partition.split('\\')[1])
            newFile=open(AUDIO_PATH+str(partition.split('\\')[1])+'.csv','a',encoding='utf-8')
            for filename in glob.iglob(partition+'/**/*.txt', recursive=True):
                # print(filename)
                # print('hello')
                with open(filename, 'r',encoding='utf-8') as file :
                    # print(filename)
                    for line in file:
                        parts = line.split()
                        try:
                            audio_file = parts[0]
                            
                        except Exception as identifier:
                            continue
                        

                        file_path = os.path.join(os.path.dirname(filename),
                                                audio_file+'.wav')                         
                        
                        try:
                            audio, sample_rate = sf.read(file_path)

                        except Exception as identifier:
                            continue
                        
                        # feats[file_path] = compute_mfcc_logFilterBanks(audio, sample_rate)
                        # utt_len[file_path] = feats[file_path].shape[0]
                        target = ' '.join(parts[1:])
                        target=target.lower()
                        # print(file_path+','+target)
                        newFile.writelines (file_path+','+target+'\n')
            newFile.close()
            # break
if __name__ == "__main__":
    change()                