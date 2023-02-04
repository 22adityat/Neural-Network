import java.util.*;
import java.io.*;
/*
* Aditya Tagore
* 12/3/2021
* 
* This class creates a simple perceptron with two hidden layers and multiple outputs
* This configuration is also known as an A-B-C-D configuration. It can perform gradient 
* descent using backpropagation to learn multiple functions such as XOR, AND and OR simultaneously. 
*
* Methods contained in file: double f(double x), 
* double randomNumGenerator(double high, double low), void randomWeightInitialization(),
* double findError(double[] trueOutput), void runInput(double[] input), void run(), 
* void runInputForTraining(double[] input, double[] truevals), void runForTraining()
* void displayNetwork(), double fDerivative(double x), void updateWeights(), 
* void train(), void main(String[] args), and void saveWeights(BufferedWriter bw)
* 
*/
public class Network 
{
   double[][] wmk;
   double[][] wkj;
   double[][] wji; 

   double[] am;
   double[] ak; 
   double[] aj;
   double[] ai;

   double[] thetak;
   double[] thetaj;
   double[] thetai;
    
   double[] psij;
   double[] omegaj;

   double[] psii;
   double[] omegai; 

   double[] psik;
   double[] omegak;  
   double[] Ti;                           
   
   double[][] outputs;

   double high;
   double low;
   int nMax;
   double errorThreshold; 
   double lambda;
   int numIterations;
   int nummLayerNodes;
   int numkLayerNodes;
   int numjLayerNodes;
   int numiLayerNodes;
   int numInputs;

   double[][] inputArray;          
   double[][] truthTable;

   double[][] pwkj;
   double[][] pwji;
   double[][] pwmk;

   int weightsaveinterval;    // Interval of iterations to save weights at
   String train;              // Whether to train or run network
   String weightsformat;      // Whether weights are to be read from a file or randomized
  
   static String controlfile;
   String weightsavefile;    // name of file to save weights to
   String weightsloadfile;   // name of file to load weights from

   /*
   * Constructor for the Network class, creating a Network object
   * and initializing the necessary instance variables.  
   */
   public Network() throws FileNotFoundException
   {
      File file = new File("//Users//adityatagore//Desktop//NeuralNets//" + controlfile); // master control file
      Scanner sc = new Scanner(file);
      nummLayerNodes = sc.nextInt();
      numkLayerNodes = sc.nextInt();
      numjLayerNodes = sc.nextInt();
      numiLayerNodes = sc.nextInt();
      high = sc.nextDouble();
      low = sc.nextDouble();
      nMax = sc.nextInt();
      errorThreshold = sc.nextDouble();
      lambda = sc.nextDouble();
      numIterations = 0;
      numInputs = sc.nextInt();
      weightsaveinterval = sc.nextInt();
      train = sc.next();
      weightsformat = sc.next();
      weightsavefile = sc.next();
      weightsloadfile = sc.next();

      inputArray = new double[numInputs][nummLayerNodes];

      for (int x = 0; x < numInputs; x++)
      {
         for (int m = 0; m < nummLayerNodes; m++)
         {
            inputArray[x][m] = sc.nextDouble();
            //System.out.print(inputArray[x][m] + " ");
         }
      }

      truthTable = new double[numInputs][numiLayerNodes];

      for (int x = 0; x < numInputs; x++)
      {
         for (int i = 0; i < numiLayerNodes; i++)
         {
            truthTable[x][i] = sc.nextDouble();
         }
      }

      outputs = new double[numInputs][numiLayerNodes]; 
      sc.close();
     
      wmk = new double[nummLayerNodes][numkLayerNodes];
      wkj = new double[numkLayerNodes][numjLayerNodes];
      wji = new double[numjLayerNodes][numiLayerNodes];

      am = new double[nummLayerNodes];
      ak = new double[numkLayerNodes];
      aj = new double[numjLayerNodes];
      ai = new double[numiLayerNodes];

      thetak = new double[numkLayerNodes];
      thetaj = new double[numjLayerNodes];
      thetai = new double[numiLayerNodes];
      
      omegak = new double[numkLayerNodes];
      psik = new double[numkLayerNodes];
      omegaj = new double[numjLayerNodes];
      psij = new double[numjLayerNodes];
      psii = new double[numiLayerNodes];
      omegai = new double[numiLayerNodes];
      Ti = new double[numiLayerNodes];
   } // public Network() throws FileNotFoundException

   /*
   * Applies the threshold function to input variable x,
   * which must be of type double. Returns the result,
   * also a double. 
   */
   public double f(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

   /*
   * Generates a random value in the range from low to high.
   * Parameters high and low must be of type double, and the 
   * return value is also a double. 
   */
   public double randomNumGenerator(double high, double low)
   {
      double x = (Math.random() * (high - low)) + low;
      return x;
   }

   /*
   * Initializes all weights (specifically, the mk
   * weights, the kj weights and the ji weights), to random numbers 
   * as determined by the randomNumGenerator function. 
   */
   public void randomWeightInitialization()
   {
      for (int m = 0; m < nummLayerNodes; m++)
      {
         for (int k = 0; k < numkLayerNodes; k++)
         {
            wmk[m][k] = randomNumGenerator(high, low);
         }
      } // for (int m = 0; m < nummLayerNodes; m++)
 
      for (int k = 0; k < numkLayerNodes; k++)
      {
         for (int j = 0; j < numjLayerNodes; j++)
         {
            wkj[k][j] = randomNumGenerator(high, low);       
         }    
      } // for (int k = 0; k < numkLayerNodes; k++)

      for (int j = 0; j < numjLayerNodes; j++)
      {
         for (int i = 0; i < numiLayerNodes; i++)    
         {
            wji[j][i] = randomNumGenerator(high, low);
         }
      } // for (int j = 0; j < numjLayerNodes; j++)
   } // public void randomWeightInitialization()

   /*
   * Determines the value of the network's error. Error
   * is determined by accumulating the squared difference 
   * between the network's output and the true output across
   * all output nodes and multiplying by 1/2. The parameter
   * trueOutput must be of type double, and the return 
   * value is also a double. 
   */
   public double findError(double[] trueOutput)
   {
      double sumDifference = 0.0;

      for (int i = 0; i < numiLayerNodes; i++)
      {
         double difference = trueOutput[i] - ai[i];
         sumDifference += difference*difference;
      } 
      
      return 0.5 * sumDifference;
   } // public double findError(double[] trueOutput)

   /*
   * Runs a single input through the network, computing
   * dot products between weights and inputs as well as 
   * applying the f function to activations. The 
   * outputs of the network are stored in ai. Parameters input and truevals
   * must be single dimensional arrays containing doubles. Omegai and psii, 
   * which are needed for backpropagation, are calculated
   * and saved to be used when weights are updated. 
   */
   public void runInputForTraining(double[] input, double[] truevals)
   {
      am = input;
      Ti = truevals;

      for (int i = 0; i < numiLayerNodes; i++)
      {
         thetai[i] = 0.0;

         for (int j = 0; j < numjLayerNodes; j++)
         {
            thetaj[j] = 0.0;
            
            for (int k = 0; k < numkLayerNodes; k++)
            {
               thetak[k] = 0.0;

               for (int m = 0; m < nummLayerNodes; m++)
               {
                  thetak[k] += am[m] * wmk[m][k];
               } // for (int m = 0; m < nummLayerNodes; m++)

               ak[k] = f(thetak[k]);
               thetaj[j] += ak[k] * wkj[k][j];
            } // for (int k = 0; k < numkLayerNodes; k++)

            aj[j] = f(thetaj[j]);
            thetai[i] +=  aj[j] * wji[j][i];  
         } // for (int j = 0; j < numjLayerNodes; j++)

         ai[i] = f(thetai[i]);
         omegai[i] = Ti[i] - ai[i];
         psii[i] = omegai[i] * fDerivative(thetai[i]);        
      } // for (int i = 0; i < numiLayerNodes; i++)   
   } // public void runInputForTraining(double[] input, double[] truevals)
 
   /*
   * Runs a single input through the network, computing
   * dot products between weights and inputs as well as 
   * applying the f function to activations. The 
   * outputs of the network are stored in ai. Parameter inputs
   * input and truevals must be a single dimensional arrays containing doubles. 
   */
  public void runInput(double[] input, double[] truevals)
  {
     am = input;
     Ti = truevals;

     for (int i = 0; i < numiLayerNodes; i++)
     {
        thetai[i] = 0.0;

        for (int j = 0; j < numjLayerNodes; j++)
        {
           thetaj[j] = 0.0;
           
           for (int k = 0; k < numkLayerNodes; k++)
           {
              thetak[k] = 0.0;

              for (int m = 0; m < nummLayerNodes; m++)
              {
                 thetak[k] += am[m] * wmk[m][k];
              } // for (int m = 0; m < nummLayerNodes; m++)

              ak[k] = f(thetak[k]);
              thetaj[j] += ak[k] * wkj[k][j];
           } // for (int k = 0; k < numkLayerNodes; k++)

           aj[j] = f(thetaj[j]);
           thetai[i] +=  aj[j] * wji[j][i];  
        } // for (int j = 0; j < numjLayerNodes; j++)

        ai[i] = f(thetai[i]);       
     } // for (int i = 0; i < numiLayerNodes; i++)   
  } // public void runInput(double[] input)

   /*
   * Runs all inputs through the network in preparation
   * for training by calling the runInputForTraining method 
   * for all the inputs. Computes the total error for all inputs
   * of the network. Prints inputs, network's outputs from run, truth table, 
   * and total error
   */
   public void runForTraining()
   {
      double totalError = 0.0;
      
      for (int x = 0; x < inputArray.length; x++)
      {
         runInputForTraining(inputArray[x], truthTable[x]);  

         for (int i = 0; i < numiLayerNodes; i++) 
         {
            outputs[x][i] = ai[i];
         } 

         double error = findError(truthTable[x]);
         totalError += error;   

      } // for (int x = 0; x < inputArray.length; x++)

      System.out.println();
      System.out.println("RUNNING... ");
      System.out.println();
      System.out.println("Total error: " + totalError + " ");
      displayNetwork();
   } // public void runForTraining()

   /*
   * Runs all inputs through the network by calling the 
   * runInput method for all the inputs. Computes the total 
   * error for all inputs of the network. Prints inputs,
   * network's outputs from run, truth table, and total error. This
   * method should not be used for training - it is for running the
   * network by itself, typically after training is over
   */
  public void run()
  {
     double totalError = 0.0;
     
     for (int x = 0; x < inputArray.length; x++)
     {
        runInput(inputArray[x], truthTable[x]);     
        
        for (int i = 0; i < numiLayerNodes; i++) 
        {
           outputs[x][i] = ai[i];
        } 

        double error = findError(truthTable[x]);
        totalError += error;   

     } // for (int x = 0; x < inputArray.length; x++)

     System.out.println();
     System.out.println("RUNNING... ");
     System.out.println();
     System.out.println("Total error: " + totalError + " ");
     displayNetwork();
  } // public void run()

   /*
   * Outputs information about the network to the terminal window.
   * Information includes network configuration, randomized weight
   * initialization values, input values, output values, and truth
   * table values
   */
   public void displayNetwork()
   {
      System.out.println("Lower bound of random numbers: " + low);
      System.out.println("Higher bound of random numbers: " + high);
      System.out.println("Number of input layer nodes: " + nummLayerNodes);
      System.out.println("Number of hidden layer 1 nodes: " + numkLayerNodes);
      System.out.println("Number of hidden layer 2 nodes: " + numjLayerNodes);
      System.out.println("Number of output nodes: " + numiLayerNodes);
      System.out.println();
      System.out.println();
      System.out.println("Inputs\t\t\tTruth Table\t\t\tOutput");
      System.out.println();

      for (int i = 0; i < inputArray.length; i++)
      {
          
         System.out.print("Image " + i);

         if (i < 10)
         {
            System.out.print("\t\t\t");
         }
         else
         {
            System.out.print("\t\t");
         }
   
         System.out.print(truthTable[i][0]);

         System.out.print("\t\t\t");
         

         System.out.print(outputs[i][0]);           

         System.out.println();
         System.out.println();
      } // for (int i = 0; i < inputArray.length; i++)
   } // public void displayNetwork()

   /*
   * Computes the derivative of the threshold function at a certain 
   * value x. Parameter x must be of type double. Return value is
   * also a double. 
   */
   public double fDerivative(double x)
   {
      double f = f(x);
      return f * (1.0 - f);
   }

   /*
   * Updates the network's weights through backpropagation-optimized gradient descent, which 
   * minimizes the error function with respect to each weight. It does
   * this by moving in the direction opposite to the gradient towards a
   * minimum. The speed of this movement is determined by the lambda 
   * instance variable. 
   */
   public void updateWeights()
   {
      for (int j = 0; j < numjLayerNodes; j++)
      {
         omegaj[j] = 0.0;

         for (int i = 0; i < numiLayerNodes; i++)
         {
            omegaj[j] += psii[i] * wji[j][i];
            wji[j][i] += lambda * aj[j] * psii[i];
         }

         psij[j] = omegaj[j] * fDerivative(thetaj[j]);
      } // for (int j = 0; j < numjLayerNodes; j++)

      for (int k = 0; k < numkLayerNodes; k++)
      {
         omegak[k] = 0.0;

         for (int j = 0; j < numjLayerNodes; j++)
         {
            omegak[k] += psij[j] * wkj[k][j];
            wkj[k][j] += lambda * ak[k] * psij[j];
         }

         psik[k] = omegak[k] * fDerivative(thetak[k]);
      } // for (int k = 0; k < numkLayerNodes; k++)

      for (int m = 0; m < nummLayerNodes; m++)
      {
         for (int k = 0; k < numkLayerNodes; k++)
         {
            wmk[m][k] += lambda * am[m] * psik[k];
         }
      } // for (int m = 0; m < nummLayerNodes; m++)
   } // public void updateWeights()       

   /*
   * Saves the current weight values to a text file specified by the user in the control file
   * (weightsavefile) where these values can later be retrieved for running or training. The 
   * BufferedWriter parameter bw must have access to an existing file for this to work. 
   */
   public void saveWeights(BufferedWriter bw) throws FileNotFoundException, IOException
   {
      for (int m = 0; m < nummLayerNodes; m++)
      {
         for (int k = 0; k < numkLayerNodes; k++)
         {
            bw.write(wmk[m][k] + " ");
         }

         bw.newLine();
      } // for (int m = 0; m < nummLayerNodes; m++)

      bw.newLine();

      for (int k = 0; k < numkLayerNodes; k++)
      {
         for (int j = 0; j < numjLayerNodes; j++)
         {
            bw.write(wkj[k][j] + " ");
         }

         bw.newLine();
      } // for (int k = 0; k < numkLayerNodes; k++)

      bw.newLine();

      for (int j = 0; j < numjLayerNodes; j++)
      {
         for (int i = 0; i < numiLayerNodes; i++)
         {
            bw.write(wji[j][i] + " ");
         }

         bw.newLine();
      } // for (int j = 0; j < numjLayerNodes; j++)
   } // public void saveWeights(BufferedWriter bw) throws FileNotFoundException, IOException

   /*
   * Runs the training algorithm on the network until either the number
   * of iterations has surpassed nMax or the network's error is less than
   * an error threshold. To train, the network iterates through each input, 
   * running the network, updating the weights through backpropagation, and calculating the network's 
   * error. 
   */
   public void train() throws FileNotFoundException, IOException
   {
      double totalError = Double.MAX_VALUE;

      while (numIterations < nMax && totalError > errorThreshold)
      {    
         totalError = 0.0;

         for (int x = 0; x < inputArray.length; x++)
         {
            runInputForTraining(inputArray[x],truthTable[x]);  
            double error = findError(truthTable[x]);
            totalError += error;

            if (totalError > errorThreshold)  
            {
               updateWeights();
            }

            if (weightsaveinterval > 0 && (numIterations % weightsaveinterval == 0))
            {
               File f = new File("//Users//adityatagore//Desktop//NeuralNets//" + weightsavefile);
               BufferedWriter bw = new BufferedWriter(new FileWriter(f));
               saveWeights(bw);
               bw.close();
            }
                                           
         } // for (int x = 0; x < inputArray.length; x++)

         if (totalError > errorThreshold) 
         {
            numIterations++; 
         } 
         if (numIterations % 50 == 0)
         {  
            System.out.println(numIterations + "\t" + totalError);
         }
         
      } // while (numIterations < nMax && totalError > errorThreshold)

      for (int x = 0; x < inputArray.length; x++)
      {
         runInput(inputArray[x], truthTable[x]); 
         for (int i = 0; i < numiLayerNodes; i++) 
         {
            outputs[x][i] = ai[i];
         }      
      } // for (int x = 0; x < inputArray.length; x++)

      //System.out.print(weightsavefile);
      
      File f = new File("//Users//adityatagore//Desktop//NeuralNets//" + weightsavefile);
      BufferedWriter bw = new BufferedWriter(new FileWriter(f));
      saveWeights(bw);
      bw.close();
      
      System.out.println();  
      System.out.println();
      System.out.println("TRAINING... ");
      System.out.println();

      if (numIterations == nMax)
      {
         System.out.println("Terminated -  nMax was reached");
      }
      else
      {
         System.out.println("Terminated - error below threshold");
      }  

      System.out.println("Total Error: " + totalError);
      System.out.println("Number of iterations: " + numIterations);
      System.out.println("Lambda: " + lambda);
      System.out.println("Error threshold: " + errorThreshold);
      System.out.println("Nmax: " + nMax);
      displayNetwork();
   } // public void train() throws FileNotFoundException, IOException
    
   
   /*
   * Main Method - instantiates a Network object. If no command line arguments are given,
   * the default control file of parameters.txt will be used. If a single command line argument
   * is provided, that argument will be used as the control file. Based on data in the control file,
   * the network can either train or test. For training, if the weightsformat, as specified in the control
   * file, is "fromfile", the network will train using weights in an existing weights file (weightsloadfile), 
   * also specified in the control file. If the weightsformat is not "fromfile", the network will train using 
   * randomly generated weights. For running, if the weightsformat, as specified in the control file, is "fromfile", 
   * the network will run using weights from an existing weights file (weightsloadfile), also specified 
   * in the control file. This might be, for example, the weights saved from the most recent training, 
   * or a preloaded set of weights in a different file. If the weightsformat is not "fromfile", the network
   * will run using a randomly generated set of weights. 
   */
   public static void main(String[] args) throws FileNotFoundException, IOException
   {
      if (args.length == 0)
      {
         controlfile = "parameters.txt";
      }
      else
      {
         controlfile = args[0];
      }
      
      Network n = new Network();
      File f1 = new File("//Users//adityatagore//Desktop//NeuralNets//" + n.weightsloadfile);
      Scanner sc1 = new Scanner(f1);

      if (n.train.equals("train"))
      {
         if (n.weightsformat.equals("fromfile"))
         {
            n.pwmk = new double[n.nummLayerNodes][n.numkLayerNodes];
            n.pwkj = new double[n.numkLayerNodes][n.numjLayerNodes];
            n.pwji = new double[n.numjLayerNodes][n.numiLayerNodes];

            for (int m = 0; m < n.nummLayerNodes; m++)
            {
               for (int k = 0; k < n.numkLayerNodes; k++)
               {
                  n.pwmk[m][k] = sc1.nextDouble();
               }
            }

            for (int k = 0; k < n.numkLayerNodes; k++)
            {
               for (int j = 0; j < n.numjLayerNodes; j++)
               {
                  n.pwkj[k][j] = sc1.nextDouble();
               }
            }
   
            for (int j = 0; j < n.numjLayerNodes; j++)
            {
               for (int i = 0; i < n.numiLayerNodes; i++)
               {
                  n.pwji[j][i] = sc1.nextDouble();
               }
            }
            
            n.wmk = n.pwmk;
            n.wkj = n.pwkj;
            n.wji = n.pwji;
            long startTime = System.currentTimeMillis();
            n.train();
            BufferedWriter bw = new BufferedWriter(new FileWriter(f1));
            n.saveWeights(bw);
            bw.close();

            long endTime = System.currentTimeMillis();
            System.out.println();
            System.out.println("Time taken to train: " + (endTime-startTime));      
         } // if (n.weightsformat.equals("fromfile"))
         else
         {
            long startTime = System.currentTimeMillis();
            n.randomWeightInitialization();
            n.train();
            long endTime = System.currentTimeMillis();
            System.out.println();
            System.out.println("Time taken to train: " + (endTime-startTime));
         } // else 
      } // if (n.train.equals("train"))
      else 
      {
         if (n.weightsformat.equals("fromfile"))
         {
            n.pwmk = new double[n.nummLayerNodes][n.numkLayerNodes];
            n.pwkj = new double[n.numkLayerNodes][n.numjLayerNodes];
            n.pwji = new double[n.numjLayerNodes][n.numiLayerNodes];

            for (int m = 0; m < n.nummLayerNodes; m++)
            {
               for (int k = 0; k < n.numkLayerNodes; k++)
               {
                  n.pwmk[m][k] = sc1.nextDouble();
               }
            }

            for (int k = 0; k < n.numkLayerNodes; k++)
            {
               for (int j = 0; j < n.numjLayerNodes; j++)
               {
                  n.pwkj[k][j] = sc1.nextDouble();
               }
            }

            for (int j = 0; j < n.numjLayerNodes; j++)
            {
               for (int i = 0; i < n.numiLayerNodes; i++)
               {
                  n.pwji[j][i] = sc1.nextDouble();
               }
            }

            n.wmk = n.pwmk;
            n.wkj = n.pwkj;
            n.wji = n.pwji;
            n.run();    
         } // if (n.weightsformat.equals("fromfile"))
         else
         {
            n.randomWeightInitialization();
            n.run();
         } // else
      } // else

      sc1.close();

   } // public static void main(String[] args) throws FileNotFoundException, IOException
} // public class Network 
