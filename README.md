A pedestrian detection demo based on YOLOX

On the basis of the original project, replace OpenCV with yolox as the target detector.

This project has re encapsulated the YOLOX tiny and YOLOX nano networks, providing a simpler calling interface.

The project utilized trained models obtained by YOLOX official on the Coco dataset. 
This demo only focuses on the category of people, so the obtained categories are filtered in the code. Of course, you can also train your own model and obtain other categories of interest.

Use Torch as the inference engine and pyqt to implement interactive pages.

