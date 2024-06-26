A pedestrian detection demo based on YOLOX

On the basis of the original project, replace OpenCV with yolox as the target detector.

This project has re encapsulated the YOLOX tiny and YOLOX nano networks, providing a simpler calling interface.

The project utilized trained models obtained by YOLOX official on the Coco dataset. 
This demo only focuses on the category of people, so the obtained categories are filtered in the code. Of course, you can also train your own model and obtain other categories of interest.

Use Torch as the inference engine and pyqt to implement interactive pages.

------------------------------------------------------------------------------------

这是一个很适合深度学习模型部署练手的小项目
可以初步了解一下GUI编程、深度学习模型部署
这个小项目大可作为宏大毕业设计前的开胃菜
共勉~