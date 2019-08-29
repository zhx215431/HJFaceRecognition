import tfcoreml
tfcoreml.convert(tf_model_path="E:/study/DL/HJFaceRecognition/project/InceptionModel_useful.pb",
                     mlmodel_path="E:/study/DL/HJFaceRecognition/project/InceptionModelTest.mlmodel",
                     output_feature_names=['output:0'],
                     input_name_shape_dict={'input2333:0': [1,64,64,3]},
                     image_input_names=['input2333:0'],
                     class_labels=['大眼女','女主','干净男','年轻老太婆','男主','神谷低配','老太婆','长嘴男','高晓松']
                     )
