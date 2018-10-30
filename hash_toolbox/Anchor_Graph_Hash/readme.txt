
Please first see demo_AGH.m to find how my codes work.

To run AGH, one needs to input anchors. In my ICML'11 paper, I used K-means clustering centers 
as anchors. If one had any sophisticated or task specific clustering algorithms, it could be better to feed 
the resulting clustering centers to anchors. Nevertheless, I found K-means anchors are sufficiently good.

Another possible issue is kernel. I used Gaussian RBF kernel throughout my paper, but any other kernels are
admittable (the clustering algorithm should be compatible with kernels). Please ask me if you do not know 
how to incorporate other kernels.

For any problem with my codes, feel free to drop me a message via wliu@ee.columbia.edu. Also, I hope you
to cite my ICML'11 paper in your publications.

Wei Liu
Sep 9, 2011

  






 