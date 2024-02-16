import kfp
from kfp import dsl

import os

dataset_name='ag_news'
task='default'
split='train[:10%]'
preprocessed_data_path='preprocessed_data.pkl'
test_data='preprocessed_data.pkl'
model_path='model'

def preprocess_data_op(dataset_name : str, task: str, split: str) :
    os.system('docker build -t maamounm/gpt2_preprocess_data:latest ./Preprocess_data')
    os.system('docker push maamounm/gpt2_preprocess_data:latest')
    return dsl.ContainerOp(
        name="Data Processing",
        image='maamounm/gpt2_preprocess_data:latest',
        arguments=[
            dataset_name,   
            task,  
            split,
        ],
    )


def fine_tune_model_op(preprocessed_data_path: str):
    os.system('docker build -t maamounm/gpt2-fine-tuning:latest ./fine_tune_model')
    os.system('docker push maamounm/gpt2-fine-tuning:latest')
    return dsl.ContainerOp(
        name="Model fine tunning",
        image='maamounm/gpt2-fine-tuning:latest',
        arguments=[
            preprocessed_data_path
        ],
    )

def evaluate_model_op(model_path:str,test_data:str):
    os.system('docker build -t maamounm/gpt2-evaluate_model:latest ./evaluate_model')
    os.system('docker push maamounm/gpt2-evaluate_model:latest')
    return dsl.ContainerOp(
        name="Model evaluation",
        image='maamounm/gpt2-evaluate_model:latest',
        
        arguments=[
            model_path,  # Or dynamically pass these values
            test_data
        ],
    )

"""
# Define component from the Docker image
@dsl.container_component
def preprocess_data_op(dataset_name : str, task: str, split: str) :
    os.system('docker build -t maamounm/gpt2_preprocess_data:latest ./Preprocess_data')
    os.system('docker push maamounm/gpt2_preprocess_data:latest')
    return  dsl.ContainerSpec(
        image='maamounm/gpt2_preprocess_data:latest',
        
        args=[
            dataset_name,   # Or dynamically pass these values
            task,  
            split,
        ],
    )



@dsl.container_component
def fine_tune_model_op(preprocessed_data_path: str):
    os.system('docker build -t maamounm/gpt2-fine-tuning:latest ./fine_tune_model')
    os.system('docker push maamounm/gpt2-fine-tuning:latest')
    return dsl.ContainerSpec(
        image='maamounm/gpt2-fine-tuning:latest',
        args=[
            preprocessed_data_path
        ],
    )

@dsl.container_component
def evaluate_model_op(model_path:str,test_data:str):
    os.system('docker build -t maamounm/gpt2-evaluate_model:latest ./evaluate_model')
    os.system('docker push maamounm/gpt2-evaluate_model:latest')
    return dsl.ContainerSpec(
        image='maamounm/gpt2-evaluate_model:latest',
        #command=['python', '-c'],
        args=[
            model_path,  # Or dynamically pass these values
            test_data
        ],
    )


"""

# Define the pipeline
@dsl.pipeline(
    name='GPT-2 Fine-Tuning Pipeline',
    description='A pipeline that fine-tunes a GPT-2 model.'
)

def gpt2_pipeline(dataset_name: str,task:str,split:str,preprocessed_data_path:str,test_data:str):
    dataset_name='ag_news'
    task='default'
    split='train[:10%]'
    preprocessed_data_path='preprocessed_data.pkl'
    test_data='preprocessed_data.pkl'
    model_path='model'
    preprocess_task = preprocess_data_op(dataset_name=dataset_name, task=task, split=split)
    fine_tune_task = fine_tune_model_op(preprocessed_data_path=preprocessed_data_path).after(preprocess_task)
    evaluate_task = evaluate_model_op(model_path=model_path, test_data=test_data).after(fine_tune_task)

# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(gpt2_pipeline, 'gpt2_pipeline.yaml')
    