import tfcoreml
tfcoreml.convert(tf_model_path="E:/study/DL/HJFaceRecognition/project/model1.pb",
                     mlmodel_path="E:/study/DL/HJFaceRecognition/project/modelTest3.mlmodel",
                     output_feature_names=['output:0'],
                     input_name_shape_dict={'input666:0': [1,64,64,3]},
                     image_input_names=['input666:0']
                     )
