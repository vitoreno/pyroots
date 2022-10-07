- models\20220630_151243.pth
    - Class Net
    - criterion = nn.CrossEntropyLoss()
    - optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model: 20220713_172832 args: Namespace(bTrain=True, bValidate=True, DATASET_PATH='C:\\Users\\vitor\\Desktop\\bin_class_65', batch_size=128, model_fn='model.pth') Accuracy of the network on the 181156 test images: 89 %

model: 20220923_171621 args: Namespace(bTrain=True, bValidate=True, DATASET_PATH='C:\\Users\\vitor\\Desktop\\Roots20220923_Subset\\bin_65', batch_size=128, model_fn='model.pth')

model: 20221003_165413 args: Namespace(bTrain=True, bValidate=True, DATASET_PATH='C:\\Users\\vitor\\Desktop\\Roots20220923_Subset\\bin_257', batch_size=64, model_fn='model.pth', img_width_height=257)

model: .\models\20221006_170210.pth args: Namespace(bTrain=True, bValidate=True, DATASET_PATH='C:\\Users\\vitor\\Desktop\\Roots20220923_Subset\\bin_257', batch_size=64, model_fn='model.pth', img_width_height=257)
