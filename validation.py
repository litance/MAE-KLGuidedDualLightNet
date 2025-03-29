# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import matplotlib.pyplot as plt
# from modelA import MobileNetLSTMSTAM
# from modelB import ESNetLSTM
# from dataload import transform
#
# # 加载模型
# modelA = MobileNetLSTMSTAM()  # 使用你的模型类名称
# modelB = ESNetLSTM()  # 使用你的模型类名称
#
# modelA.load_state_dict(torch.load('model/modelA.pth'))
# modelB.load_state_dict(torch.load('model/modelB.pth'))
#
# modelA.eval()
# modelB.eval()
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# modelA = modelA.to(device)
# modelB = modelB.to(device)
#
# test_dataset = datasets.ImageFolder('../dataset/test_dataset/asl_dataset', transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # 验证函数
# def validate(model, loader):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     return accuracy
#
# # 计算准确率
# accuracyA = validate(modelA, test_loader)
# accuracyB = validate(modelB, test_loader)
#
# # 绘制结果
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.bar(['Model A'], [accuracyA], color='blue')
# plt.title('Model A Accuracy')
# plt.ylim(0, 100)
#
# plt.subplot(1, 2, 2)
# plt.bar(['Model B'], [accuracyB], color='green')
# plt.title('Model B Accuracy')
# plt.ylim(0, 100)
#
# plt.tight_layout()
# plt.savefig('validation_results.png')
# plt.show()
#
# print(f'Model A Accuracy: {accuracyA:.2f}%')
# print(f'Model B Accuracy: {accuracyB:.2f}%')
