#pragma once

#include <JuceHeader.h>
//#include <iostream>
//#include <vector>
//#include <set>
//#include <iterator>
//#include <algorithm>
#include "../SenselWrapper/SenselWrapper.h"
//==============================================================================
/*
    This component lives inside our window, and this is where you should put all
    your controls and content.
*/
class MainComponent : public juce::AudioAppComponent,
                    public juce::Slider::Listener,
                    public juce::Timer,
                    public juce::HighResolutionTimer
{
public:
    //==============================================================================
    MainComponent();
    ~MainComponent() override;

    //void mouseDrag(const MouseEvent& e) override;
    //void mouseDown(const MouseEvent& e) override;
    //void mouseUp(const MouseEvent& e) override;
    //////void mouseExit(const MouseEvent& e) override;


    //==============================================================================
    void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override;
    void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) override;
    void releaseResources() override;


    void timerCallback() override;
    void hiResTimerCallback() override;

    //==============================================================================
    void paint(juce::Graphics& g) override;
    void resized() override;
    void sliderValueChanged(Slider* slider) override;


    //==============================================================================
    double clamp(double input, double min, double max);




private:
    //==============================================================================
    // Your private member variables go here...

    /// start global parameters

    // Graphics Stuff:
    bool graphicsToggle = true;

    // Sensel Stuff
    OwnedArray<Sensel> sensels;
    int amountOfSensels = 1; 

    // SLIDER STUFF

    Slider frequencySlider;
    Label frequencyLabel;

    Slider sig_0_Slider;
    Label sig_0_Label;

    Slider sig_1_Slider;
    Label sig_1_Label;

    // FILE READING  STUFF
    std::vector<int> locsDo_X;
    std::vector<int> locsDo_Y;

    // GENERAL PARAMETERS
    double fs;
    //float gain = 1000000000;
    float gain = 5000;
    //float gain = 1000;
    float t = 0;
    float preOutput;
    double k; // time step in Finite Difference scheme


    // BOWING MODEL PARAMETERS
    double mu_C = 0.3;
    double mu_S = 0.8;
    //double fN = 0.5;
    double fN = 1;
    //double fN = 4;
    //double fC = fN * mu_C;
    //double fS = fN * mu_S;
    double fC;
    double fS;
    double vB = 0.1;
    double vS = 0.1;
    double s0 = 1e4;
    double s1 = 0.001 * sqrt(s0);
    //double s1 = 0.0;
    double s2 = 0.4;
    double s3_fac = 0;
    //double s3 = 0.00 * fN;
    double s3;
    double w_rnd;
    //double w_rnd_vec = -1 + (1 - (-1)).*rand(NF, 1); // generate rand value at each t iteration
    //double z_ba = 0.7 * (mu_C * fN) / s0;
    double z_ba;
    double tol = 0.0000001;

    // STRING PARAMETERS
    double rho = 7850; // [kg / m ^ 3]
    double radius = 5e-4;
    double A = double_Pi * radius * radius; // [m ^ 2]
    //double L = 0.3624;
    double L = 0.4978;
    double E = 2e11;
    double I = double_Pi * radius * radius * radius * radius / 4;
    double K = sqrt(E * I / (rho * A));
    //double f0 = 116.54; // A2#
    //double f0 = 233.08; // A3#
    double f0 = 220.00; // A3
    //double f0 = 369.99; // F#
    double c = 2 * f0 * L;

    double* lossFreqs = new double[2]{ 100, 3000 };
    double* lossT60s = new double[2]{ 3, 1 };
    double zeta1;
    double zeta2;
    double sig0;
    double sig1;

    double h; // spatial step in FD scheme
    double hFact = 1.0;
    double N; // no of spatial points to discretize string 

    // STRING CONNECTION:
    double x_conn_string = 0.7674; // in percentage

    std::vector<double> I_S; // interpolant grid
    std::vector<double> J_S; // spreading function grid

    // PLATE PARAMETERS
    double L_plate = 0.526; // working with real dimensions
    double rho_plate = 400; // Density of violin wood - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002554
    double H_plate = 0.001; // [m] check: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0002554 // very thin ! 
    double E_plate = 10000000000; // [Pa] [N/m^2]
    double nu_plate = 0.3;
    double D_plate = E_plate * H_plate * H_plate * H_plate / (12 * (1 - nu_plate * nu_plate));
    double K_plate = sqrt(D_plate / (rho_plate * H_plate));

    double* lossFreqs_plate = new double[2]{ 100, 3000 };
    double* lossT60s_plate = new double[2]{ 3, 1 };
    double zeta1_plate;
    double zeta2_plate;
    double sig0_plate;
    double sig1_plate;

    double h_plate; // spatial step in FD scheme
    double hFact_plate = 1.0;
    double N_plate; // no of spatial points to discretize string 
    int Nx_plate;
    int Ny_plate;

    // PLATE CONNECTION:
    //double x_conn_plate = 0.42; // absolute values
    //double y_conn_plate = 0.2953;

    // I think these are switched.. 
    double x_conn_plate = 0.4; // absolute values
    double y_conn_plate = 0.3;

    int l_inp_plate; // -1 for alignment with Matlab
    int m_inp_plate; // -1 for alignment with Matlab
    int l_inp_plate_plus1;
    int m_inp_plate_plus1;

    std::vector<double> I_P; // interpolant grid
    std::vector<double> J_P; // spreading function grid



    //double freq = 300; // initial string freq
    //double c = sqrt(T / (rho * H));
    //double gamma = freq * 2 * L; // gamma param in the PDE (wave speed)
    //double lossFreqs = [100, 4; 5000, 2]; // w1, T60(w1) w2, T60(w2)


    //double* lossFreqs = new double[2]{ 100, 5000 }; 
    //double* lossT60s = new double[2]{ 4, 2 }; 

    // PLATE OUTPUT:
    double x_plate_out = 0.43; // absolute value
    double y_plate_out = 0.15; // absolute value


    // BOWING POSITION INIT

    /// Parameters you may want to modulate:
    double bp = 0.29; // in percentage
    int bP;
    double alpha_bow;

    std::vector<double> I_B; // interpolant grid for bowing pos
    std::vector<double> J_B; // spreading function grid for bowing pos

    int l_inp;
    int l_inp_plus1;
    int m_inp;
    int m_inp_plus1;
    double alpha_x_inp;
    double alpha_y_inp;



    // Constants for update eq
    double C1;
    double C2;
    double C3;
    double C4;

    /// end global parameters




    // pointers to STRING states
    std::vector<double*> u;

    // states
    std::vector<std::vector<double>> uVecs;

    double* uTmp = nullptr;


    // pointers to PLATE states
    std::vector<double*> w;

    // states
    std::vector<std::vector<double>> wVecs;

    double* wTmp = nullptr;


    // INITIALIZE PARAMS FOR NEWTON-RAPHSON

    double z_prev;
    double r_last;
    double r_prev;
    double a_last;
    double a_prev;

    double vrel_last;
    double z_last;

    double F_last;

    // NEWTON-RAPHSON SOLVER
    double z_ss;
    double alpha_fct;
    double g1;
    double g2;
    double z_ss_d_vrel;
    double alpha_fct_d_vrel;
    double alpha_fct_d_z;
    double r_last_d_vrel;
    double r_last_d_z;
    double g1_d_vrel;
    double g1_d_z;
    double g2_d_vrel;
    double g2_d_z;
    double fac_1_over_det;
    double A_mat_new;
    double B_mat_new;
    double C_mat_new;
    double D_mat_new;
    double vrel;
    double z;
    double r;
    double a;
    double Fbow;

    // CONNECTION FORCE
    double F;


    //std::vector<double> uPrev; // prev values of string displacement
    //std::vector<double> u; // prev values of string displacement
    //std::vector<double> uNext; // prev values of string displacement
    std::vector<double> I_grid; // interpolant grid
    std::vector<double> J_grid; // spreading function grid
    std::vector<double> output_interim; // spreading function grid

    //std::vector<double*> I_grid;
    //std::vector<double*> J_grid; // spreading function grid


    /// buffer related parameters
    std::vector<double> myBuffer;

    // Mouse control parameters
    float opacity = 0.0;
    int mouseX;
    int mouseY;
    int mouseX_up;
    int mouseY_up;
    int mouseX_down;
    int mouseY_down;
    //float FB_var;
    double fN_var = fN;
    double vB_var = vB;
    double fC_var;
    double fS_var;
    double z_ba_var;
    double s3_var;
    float x_inp_var;
    float y_inp_var;
    float x_out_var;
    float y_out_var;
    float x_var = 0;
    float y_var = 0;
    float xPrev = 0;
    float yPrev = 0;
    float resultant = 0;
    float resultantPrev = 0;
    float vel;


    std::deque<double> x_positions;
    std::deque<double> y_positions;



    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MainComponent)
};
