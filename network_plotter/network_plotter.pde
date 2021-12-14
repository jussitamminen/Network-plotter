import netcdf.*;

PDataset data;
float[] bias_1,bias_2,bias_3,bias_4,bias_5;
float[][] weight_1,weight_2,weight_3,weight_4,weight_5;
ArrayList<float[][]> weights = new ArrayList<float[][]>();
ArrayList<float[]> biases = new ArrayList<float[]>();
float min_bias, min_weight,max_bias,max_weight,abs_max_bias,abs_max_weight;
float point_size,point_color, line_size, line_color;
float start;
float duration;

float a = 0.0;
float s = 0.0;
 void setup() {
  size(1000, 700);


  
  data = new PDataset(this);
  String filename = dataPath("dqn_training_DQN_general_1638201746_weights.hdf5");
  data.openFile(filename);

 
  data.loadData("dense/dense/bias:0");
  data.loadData("dense_1/dense_1/bias:0");
  data.loadData("dense_2/dense_2/bias:0");
  data.loadData("dense_3/dense_3/bias:0");
  data.loadData("dense_5/dense_5/bias:0");
  
  data.loadData("dense/dense/kernel:0");
  data.loadData("dense_1/dense_1/kernel:0");
  data.loadData("dense_2/dense_2/kernel:0");
  data.loadData("dense_3/dense_3/kernel:0");
  data.loadData("dense_5/dense_5/kernel:0");
  
  biases.add(bias_1 = data.get1DFloatArray("dense/dense/bias:0"));
  biases.add(bias_2 = data.get1DFloatArray("dense_1/dense_1/bias:0"));
  biases.add(bias_3 = data.get1DFloatArray("dense_2/dense_2/bias:0"));
  biases.add(bias_4 = data.get1DFloatArray("dense_3/dense_3/bias:0"));
  biases.add(bias_5 = data.get1DFloatArray("dense_5/dense_5/bias:0"));
  
  weights.add(weight_1 = data.get2DFloatArray("dense/dense/kernel:0"));
  weights.add(weight_2 = data.get2DFloatArray("dense_1/dense_1/kernel:0"));
  weights.add(weight_3 = data.get2DFloatArray("dense_2/dense_2/kernel:0"));
  weights.add(weight_4 = data.get2DFloatArray("dense_3/dense_3/kernel:0"));
  weights.add(weight_5 = data.get2DFloatArray("dense_5/dense_5/kernel:0"));
  
  //print(weight_1[0].length);
  print(weights.get(0)[0][0],"\n");
  print(weights.get(0).length,"\n");
  data.close();

  noLoop();
}

void draw() {
  background(255);
  


  
  fill(0);
  stroke(0);
  
  // Draw plot title
  textSize(18);
  textAlign(CENTER, CENTER);
  //text("Waves, waves, waves", width/2, 20);
  text(width, width/2, 20);

  // Draw y-axis
  line(50, 50, 50, height - 50);
  line(width - 50, 50, width - 50, height - 50);
  for (int y = -100; y <= 100; y += 50) {
    line(width - 50, height/2 + y, width - 60, height/2 + y);
    line(50, height/2 + y, 60, height/2 + y);
  }

  // Draw y labels
  textSize(12);
  textAlign(CENTER, CENTER);
  text("1", 40, height/2 - 100);
  text("0.5", 38, height/2 - 50);
  text("0", 40, height/2);
  text("-0.5", 34, height/2 + 50);
  text("-1", 36, height/2 + 100);

  // Draw y title
  pushMatrix();
  textAlign(CENTER, BOTTOM);
  translate(20, height/2);
  rotate(-HALF_PI);
  text("H1 bias_1 (E-19)", 0, 0);
  popMatrix();

  // Draw x-axis
  line(50, 50, width - 50, 50);
  line(50, height - 50, width - 50, height - 50);
  for (int x = 50; x <= width - 50; x += 100) {
    line(x, height - 50, x, height - 60);
    line(x, 50, x, 60);
  }

  // Draw x labels
  text(String.format("+%1.8E", start), width - 110, height - 10);
  textAlign(CENTER, CENTER);
  for (int i = 0; i < 10; i++) {
    int t = int(i * duration / 9);
    text(t, 50 + i*100, height - 40);
  }

  // Draw x title
  textAlign(CENTER, BOTTOM);
  text("GPS Time (s)", width/2, height - 10);


  
  
//--- calculate min and max values for biases and weights ---///
  for (int layer = 0; layer < biases.size(); layer++){
    
    float current_max = max(biases.get(layer));
    float current_min = min(biases.get(layer));
    if(current_max > max_bias){
      max_bias = current_max;
    }
    if(current_min < min_bias){
      min_bias = current_min;
    }
  }
  abs_max_bias = max(abs(min_bias),abs(max_bias));
  print("max bias is: ",max_bias,"\n");
  print("Min bias is: ",min_bias,"\n");
  print("Abs max bias is: ",abs_max_bias,"\n");
  
  for (float[] v : weights.get(4)){
    print("V is: ",v[0],"\n");
  }  
  
  
  
//--- Node plotting with weight start ---///
  pushMatrix();
  strokeWeight(2);
  stroke(0);
  scale(1, -1);
  translate(0, -height);
  ArrayList<PVector[]> points = new ArrayList<PVector[]>();
  

  // Input layer is plottet with small black dots
  //PVector[] node0 = new PVector[weights.get(weights.size()-1)[0].length];
  PVector[] node0 = new PVector[weights.get(0).length];
  
  //for (int i = 0; i < weights.get(weights.size()-1)[0].length; i++) {
  for (int i = 0; i < weights.get(0).length; i++) {
      //float y = map(i * (float)height/weights.get(weights.size()-1)[0].length, 0, height, 50, height - 50);
      float y = map(i * (float)height/weights.get(0).length, 0, height, 50, height - 50);
      float x = map(0, 0, width, 50, width - 50);
      float r = 50+(weights.size()-1)*150;
      point(x, y);
      node0[i] = new PVector(x, y);
      //print("points size is: ",points.size(),"\n");
  }
  points.add(node0);
  print("Node0 size is: ",node0.length,"\n");
  //print("points size is: ",points.size(),"\n");
  for (int layer = 1; layer < biases.size(); layer++){
    

    PVector[] node = new PVector[biases.get(layer).length];
    for (int i = 0; i < biases.get(layer).length; i++) {
      //print("Bias: ",biases.get(layer)[i]);
      
      point_size = map(abs(biases.get(layer)[i]),0, abs_max_bias,2,15);
      point_color = map(biases.get(layer)[i],min_bias, max_bias,0,1);
      strokeWeight(point_size);
      stroke(255*point_color, 0, 255*(1-point_color));
      float y = map(i * (float)height/biases.get(layer).length, 0, height, 50, height - 50);
      float x = map(layer * (float)width/(biases.size()-1), 0, width, 50, width - 50);
      float r = 50+layer*150;
      
      point(x, y);
      node[i] = new PVector(x, y);
      //print("points size is: ",points.size(),"\n");
    }
    points.add(node);
  }

  popMatrix();
  //--- Node plotting with weight end ---///
  
  //--- Node plotting with weight start ---///
  //pushMatrix();
  //strokeWeight(5);
  
  //scale(1, -1);
  //translate(0, -height);
  //ArrayList<PVector[]> points = new ArrayList<PVector[]>();
  ////print("points size is: ",points.size(),"\n");
  //for (int layer = 0; layer < weights.size()-1; layer++){
  //  stroke(0, 0, 255);

  //  PVector[] node = new PVector[weights.get(layer).length];
  //  for (int i = 0; i < weights.get(layer).length; i++) {
  //    float y = map(i * (float)height/weights.get(layer).length, 0, height, 50, height - 50);
  //    float x = map(layer * (float)width/(weights.size()-1), 0, width, 50, width - 50);
  //    float r = 50+layer*150;
      
  //    point(x, y);
  //    node[i] = new PVector(x, y);
  //    //print("points size is: ",points.size(),"\n");
  //  }
  //  points.add(node);
  //}
  //stroke(255, 0, 0);
  //PVector[] node = new PVector[weights.get(weights.size()-1)[0].length];
  //for (int i = 0; i < weights.get(weights.size()-1)[0].length; i++) {
  //    float y = map(i * (float)height/weights.get(weights.size()-1)[0].length, 0, height, 50, height - 50);
  //    float x = map((weights.size()-1) * (float)width/(weights.size()-1), 0, width, 50, width - 50);
  //    float r = 50+(weights.size()-1)*150;
  //    point(x, y);
  //    node[i] = new PVector(x, y);
  //    print("points size is: ",points.size(),"\n");
  //}
  //points.add(node);
  //popMatrix();
  ///--- Node plotting with weight end ---///
  
  
  //print("points size is: ",points.size(),"\n");
  //print(points.get(4));
  //float m = map(value, 0, 100, 0, width)
  
 LinePlot(points,weights);


}

void LinePlot(ArrayList<PVector[]> points,ArrayList<float[][]> weights ) {
   ///--- Line plotting start ---///
  stroke(255, 0, 0);
  pushMatrix();
  strokeWeight(1);
  scale(1, -1);
  translate(0, -height);
  for (int layer = 0; layer < points.size()-1; layer++) {
    for (int start = 0; start < points.get(layer).length; start++) {
      for (int end = 0; end < points.get(layer+1).length; end++) {
        //print("Layer : ",layer,"/n");
        //print(weights.get(layer)[start][end]);
        //print(points.get(layer)[start]);
        
        line_size = map(abs(weights.get(layer)[start][end]),0, 5,0,2);
        line_color = map(weights.get(layer)[start][end],-5, 5,0,1);
        strokeWeight(line_size);
        stroke(255*line_color, 0, 255*(1-line_color));
      
        
        float start_x = points.get(layer)[start].x;
        float start_y = points.get(layer)[start].y;
        float end_x = points.get(layer+1)[end].x;
        float end_y = points.get(layer+1)[end].y;
        line(start_x, start_y, end_x, end_y);
      }
    }
  }
  popMatrix();
  ///--- Line plotting end ---///
  

  
}
