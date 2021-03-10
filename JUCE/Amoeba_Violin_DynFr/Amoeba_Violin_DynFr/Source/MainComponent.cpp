#include "MainComponent.h"

//==============================================================================
MainComponent::MainComponent()
{
    // Make sure you set the size of the component after
    // you add any child components.
    setSize(800, 600);

    bool val = (juce::RuntimePermissions::isRequired(juce::RuntimePermissions::recordAudio)
        && !juce::RuntimePermissions::isGranted(juce::RuntimePermissions::recordAudio));

    // Some platforms require permissions to open input channels so request that here
    if (juce::RuntimePermissions::isRequired(juce::RuntimePermissions::recordAudio)
        && !juce::RuntimePermissions::isGranted(juce::RuntimePermissions::recordAudio))
    {
        juce::RuntimePermissions::request(juce::RuntimePermissions::recordAudio,
            [&](bool granted) { setAudioChannels(granted ? 2 : 0, 2); });
    }
    else
    {

        // Specify the number of input and output channels that we want to open
        setAudioChannels(2, 2);
    }



    // Sliders !
    addAndMakeVisible(frequencySlider);
    frequencySlider.setRange(50.0, 750.0);          // [1]
    frequencySlider.setTextValueSuffix(" Hz");     // [2]
    frequencySlider.addListener(this);             // [3]

    addAndMakeVisible(frequencyLabel);
    frequencyLabel.setText("Frequency", juce::dontSendNotification);
    frequencyLabel.attachToComponent(&frequencySlider, true); // [4]

    frequencySlider.setValue(220.0); // [5]

    addAndMakeVisible(sig_0_Slider);
    sig_0_Slider.setRange(0.0, 10.0);          // [1]
    sig_0_Slider.setTextValueSuffix(" [-]");     // [2]
    sig_0_Slider.addListener(this);             // [3]

    addAndMakeVisible(sig_0_Label);
    sig_0_Label.setText("Freq Independent Damping", juce::dontSendNotification);
    sig_0_Label.attachToComponent(&sig_0_Slider, true); // [4]

    sig_0_Slider.setValue(0.0); // [5]

    addAndMakeVisible(sig_1_Slider);
    sig_1_Slider.setRange(0.0, 0.5);          // [1]
    sig_1_Slider.setTextValueSuffix(" [-]");     // [2]
    sig_1_Slider.addListener(this);             // [3]

    addAndMakeVisible(sig_1_Label);
    sig_1_Label.setText("Freq Independent Damping", juce::dontSendNotification);
    sig_1_Label.attachToComponent(&sig_1_Slider, true); // [4]

    sig_1_Slider.setValue(0.0); // [5]

}

MainComponent::~MainComponent()
{
    // This shuts down the audio device and clears the audio source.
    shutdownAudio();
}

//==============================================================================
// Other functions

double linearMapping(float rangeIn_top, float rangeIn_bottom, float rangeOut_top, float rangeOut_bottom, float value) {
    double newValue = rangeOut_bottom + ((rangeOut_top - rangeOut_bottom) * (value - rangeIn_bottom) / (rangeIn_top - rangeIn_bottom));
    return newValue;
}

template <typename T> int sign(T val) { // I'm not super sure what this template thing does.. just took it from a stackoverflow thread.
    return (T(0) < val) - (val < T(0));
}


std::vector<double> hamming(int N) { // Hamming distribution vector
    double alpha = 0.54;
    double beta = 1 - alpha;

    std::vector<double> h;
    h.resize(N);

    for (int i = 0; i < N; ++i) {
        h[i] = alpha - beta * cos(2 * double_Pi * i / (N - 1));
    }

    return h;
}

std::vector<double> hann(int N) { // Hann distribution vector
    double alpha = 0.5;
    double beta = 1 - alpha;

    std::vector<double> h;
    h.resize(N);

    for (int i = 0; i < N; ++i) {
        h[i] = alpha - beta * cos(2 * double_Pi * i / (N - 1));
    }

    return h;
}

double MainComponent::clamp(double in, double min, double max)
{
    if (in > max)
        return max;
    else if (in < min)
        return min;
    else
        return in;
}





void MainComponent::sliderValueChanged(Slider* slider)
{
    //if (slider == &slider1)
    //{
    //    gain = slider1.getValue();
    //}

    //if (slider == &slider2)
    //{
    //    frequency = slider2.getValue();
    //}
    //std::cout << gain << std::endl; // Delete this line afterwards


    if (slider == &frequencySlider)
    {
        //f0 = frequencySlider.getValue();
        //c = 2 * f0 * L;
    }
    else if (slider == &sig_0_Slider)
    {
        sig0_plate = sig_0_Slider.getValue();
        Logger::getCurrentLogger()->outputDebugString(String(sig0_plate));
    }
    else if (slider == &sig_1_Slider)
    {
        sig1_plate = sig_1_Slider.getValue();
        Logger::getCurrentLogger()->outputDebugString(String(sig1_plate));
    }

}


//==============================================================================
void MainComponent::prepareToPlay(int samplesPerBlockExpected, double sampleRate) // how do I change the sampleRate .. ?
{
    // This function will be called when the audio device is started, or when
    // its settings (i.e. sample rate, block size, etc) are changed.

    // You can use this function to initialise any resources you might need,
    // but be careful - it will be called on the audio thread, not the GUI thread.

    // For more details, see the help for AudioProcessor::prepareToPlay()


    // ADD SENSEL
    for (int i = 0; i < amountOfSensels; ++i)
    {
        sensels.add(new Sensel(i)); // chooses the device in the sensel device list
        //std::cout << "Sensel added" << std::endl;
        Logger::getCurrentLogger()->outputDebugString("Sensel added");
    }

    // start the hi-res timer
    if (sensels.size() != 0)
    {
        if (sensels[0]->senselDetected)
        {
            HighResolutionTimer::startTimer(1000.0 / 150.0); // 150 Hz
            //HighResolutionTimer::startTimer(1000.0 / 10.0); // 150 Hz
            //HighResolutionTimer::startTimer(1000.0 / 1.0); // 1 Hz ?
        }
    }

    // Start timer for repaint
    Timer::startTimerHz(17);



    // Sampling rate and time step
    fs = sampleRate;
    k = 1 / fs;


    // STRING STUFF

    //zeta1 = pow(2 * double_Pi * lossFreqs[0], 2) / (c * c);
    //zeta2 = pow(2 * double_Pi * lossFreqs[1], 2) / (c * c);

    //zeta1 = (-c * c + sqrt(c * c * c * c + 4 * K * K * pow((2 * double_Pi * lossFreqs[0]), 2))) / (2 * K * K);
    //zeta2 = (-c * c + sqrt(c * c * c * c + 4 * K * K * pow((2 * double_Pi * lossFreqs[1]), 2))) / (2 * K * K);

    //sig0 = 6 * log(10) * (-zeta2 / lossT60s[0] + zeta1 / lossT60s[1]) / (zeta1 - zeta2);
    //sig1 = 6 * log(10) * (1 / lossT60s[0] - 1 / lossT60s[1]) / (zeta1 - zeta2);

    //sig0 = 0.1;
    //sig1 = 0.005;


    sig0 = 0.0;
    sig1 = 0.0;

    k = 1 / fs;

    h = sqrt((c * c * k * k + 4 * sig1 * k + sqrt(pow((c * c * k * k + 4 * sig1 * k),2) + 16 * K * K * k * k)) / 2);

    //h = sqrt(c * c * k * k + 4 * sig1 * k + sqrt(c * c * k * k + 4 * sig1 * k));
    //h = h * hFact;
    N = floor(L / h);
    if (N > 30)
    {
        N = 30;
    }
    h = L / N;


    // PLATE STUFF

    sig0_plate = 0.0;
    //sig1_plate = 0.0;
    sig1_plate = 0.05;
    //sig0_plate = 2.0;
    //sig1_plate = 0.05; 

    h_plate = 2 * sqrt(k * (sig1_plate + sqrt(K_plate * K_plate + sig1_plate * sig1_plate))); // Dimensional case !
    N_plate = floor(L_plate / h_plate);
    if (N_plate > 35)
    {
        N_plate = 35;
    }
    h_plate = L_plate / N_plate;


    // Reading locations file..


    File resourceFile("C:\\Marius\\Amoeba_Violin_DynFr\\Amoeba_Violin_DynFr\\x_locs_Matlab_N35.txt");

    if (!resourceFile.existsAsFile())
    {
        DBG("File doesn't exist ...");
    }

    FileInputStream input(resourceFile);

    if (!input.openedOk())
    {
        DBG("Failed to open file");
        // ... Error handling here
    }

    while (!input.isExhausted())
    {
        String singleLine = input.readNextLine();
        int val = singleLine.getIntValue();
        locsDo_X.push_back(val); 
        //int ana = 5;
        // ... Do something with each line
    }

    File resourceFile_Y("C:\\Marius\\Amoeba_Violin_DynFr\\Amoeba_Violin_DynFr\\y_locs_Matlab_N35.txt");

    if (!resourceFile_Y.existsAsFile())
    {
        DBG("File doesn't exist ...");
    }

    FileInputStream input_Y(resourceFile_Y);

    if (!input_Y.openedOk())
    {
        DBG("Failed to open file");
        // ... Error handling here
    }

    while (!input_Y.isExhausted())
    {
        String singleLine = input_Y.readNextLine();
        int val = singleLine.getIntValue();
        locsDo_Y.push_back(val); 
        //int ana = 5;
        // ... Do something with each line
    }


    Logger::getCurrentLogger()->outputDebugString("N: (" + String(N) + ")");
    Logger::getCurrentLogger()->outputDebugString("N_plate: (" + String(N_plate) + ")");

    // DISTRIBUTIONS FOR CONNECTIONS
    // PLATE:

    x_conn_plate = x_conn_plate / L_plate; // ratio
    y_conn_plate = y_conn_plate / L_plate; // ratio

    l_inp_plate = floor(x_conn_plate * N_plate - 1); // -1 for alignment with Matlab
    m_inp_plate = floor(y_conn_plate * N_plate - 1); // -1 for alignment with Matlab
    l_inp_plate_plus1 = l_inp_plate + 1;
    m_inp_plate_plus1 = m_inp_plate + 1;
    double alpha_x_inp = x_conn_plate * N_plate - 1 - l_inp_plate;
    double alpha_y_inp = y_conn_plate * N_plate - 1 - m_inp_plate;

    I_P.resize(N_plate * N_plate, 0);
    J_P.resize(N_plate * N_plate, 0);

    //I_P[l_inp * N_plate + m_inp] = (1 - alpha_x_inp) * (1 - alpha_y_inp); // l_inp*N+m_inp is equivalent to array[l_inp][m_inp] if it would be 2D
    //I_P[l_inp_plus1 * N_plate + m_inp] = alpha_x_inp * (1 - alpha_y_inp);
    //I_P[l_inp * N_plate + m_inp_plus1] = (1 - alpha_x_inp) * alpha_y_inp;
    //I_P[l_inp_plus1 * N_plate + m_inp_plus1] = alpha_x_inp * alpha_y_inp;

    I_P[m_inp_plate + l_inp_plate * N_plate] = (1 - alpha_x_inp) * (1 - alpha_y_inp); // m_inp+l_inp*N is equivalent to array[l_inp][m_inp] if it would be 2D
    I_P[m_inp_plate_plus1 + l_inp_plate * N_plate] = (1 - alpha_x_inp) * alpha_y_inp;
    I_P[m_inp_plate + l_inp_plate_plus1 * N_plate] = alpha_x_inp * (1 - alpha_y_inp);
    I_P[m_inp_plate_plus1 + l_inp_plate_plus1 * N_plate] = alpha_x_inp * alpha_y_inp;


    for (int i = 0; i < I_P.size(); ++i)
    {
        J_P[i] = I_P[i] * (1/(h_plate * h_plate)); // speed up: keep divisions out of loop ! 
    }

    //double I_P_J_P = 0;
    //int idx_x;
    //for (int iX = 2; iX < N_plate - 2; ++iX)
    //{
    //    for (int iY = 2; iY < N_plate - 2; ++iY)
    //    {
    //        idx_x = iY + (iX)*N_plate;

    //        I_P_J_P = I_P_J_P + I_P[idx_x] * J_P[idx_x];
    //    }
    //}

    // STRING:

    // int conn_point_idx = floor(x_conn_string * N); // Are these the same in C++ as Matlab ? what about the indexing difference ? i.e. start at 0 vs 1 ? 
    int conn_point_idx = floor(x_conn_string * N - 1) ; // Aligned with Matlab.. 
    double alph = x_conn_string * N - 1 - conn_point_idx;

    I_S.resize(N, 0);
    J_S.resize(N, 0);

    int conn_point_idx_m1 = conn_point_idx - 1;
    int conn_point_idx_p1 = conn_point_idx + 1;
    int conn_point_idx_p2 = conn_point_idx + 2;

    I_S[conn_point_idx_m1] = (alph * (alph - 1) * (alph - 2)) / -6.0;
    I_S[conn_point_idx] = ((alph - 1) * (alph + 1) * (alph - 2)) / 2.0;
    I_S[conn_point_idx_p1] = (alph * (alph + 1) * (alph - 2)) / -2.0;
    I_S[conn_point_idx_p2] = (alph * (alph + 1) * (alph - 1)) / 6.0;

    for (int i = 0; i < I_S.size(); ++i)
    {
        J_S[i] = I_S[i] * (1 / h); // speed up: keep divisions out of loop ! 
    }


    // Bow interpolation and spreading function init:
    I_B.resize(N, 0);
    J_B.resize(N, 0);

    // PLATE OUTPUT
    x_plate_out = x_plate_out / L_plate;
    y_plate_out = y_plate_out / L_plate;

    int l_plate_out = floor(x_plate_out * N_plate - 1); // -1 for alignment with Matlab 
    int m_plate_out = floor(y_plate_out * N_plate - 1); // -1 for alignment with Matlab 
    double alpha_x_plate_out = x_plate_out * N_plate - 1 - l_plate_out;
    double alpha_y_plate_out = y_plate_out * N_plate - 1 - m_plate_out;


    // INITIALIZE STRING STATE VECTORS
    uVecs.reserve(3);

    for (int i = 0; i < 3; ++i)
        uVecs.push_back(std::vector<double>(N, 0));

    u.resize(3);

    for (int i = 0; i < u.size(); ++i)
        u[i] = &uVecs[i][0];


    // INITIALIZE PLATE STATE VECTORS
    wVecs.reserve(3);

    for (int i = 0; i < 3; ++i)
        wVecs.push_back(std::vector<double>(N_plate * N_plate, 0));

    w.resize(3);

    for (int i = 0; i < w.size(); ++i)
        w[i] = &wVecs[i][0];
    



    // OUTPUT SIZE
    //output_interim.resize(samplesPerBlockExpected, 0); // what is this ? .. 

    ////FB_var = FB_base;
    ////fN_var = 0.000000001;
    //fN_var = 0.0000000001;
    ////fN_var = fN;
    ////vB_var = 0.0;
    //vB_var = vB;
    //fS_var = fN_var * mu_S;
    //fC_var = fN_var * mu_C;
    //z_ba_var = 0.7 * (mu_C * fN_var) / s0;
    //s3_var = s3_fac * fN;


    // BOWING STUFF:
    fN_var = fN;
    vB_var = vB;
    //fN_var = 0.0;
    //vB_var = 0.0;
    fS_var = fN_var * mu_S;
    fC_var = fN_var * mu_C;
    z_ba_var = 0.7 * (mu_C * fN_var) / s0;
    s3_var = s3_fac * fN;

    //// What are these for ?  
    x_inp_var = bp; // bowing pos in percentage
    y_inp_var = 0.5;

    //x_out_var = 0.7;
    //y_out_var = 0.3;



    // INITIALIZE PARAMS FOR NEWTON-RAPHSON

    z_prev = 0;
    r_last = vB_var;
    r_prev = vB_var;
    a_last = r_last;
    a_prev = r_prev;

    vrel_last = -vB_var;
    z_last = 0;

    F_last = 0;


}

void MainComponent::getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)
{
    // Your audio-processing code goes here!

    // For more details, see the help for AudioProcessor::getNextAudioBlock()

    // Right now we are not producing any data, in which case we need to clear the buffer
    // (to prevent the output of random noise)
    bufferToFill.clearActiveBufferRegion(); // clears input buffer..

    float* const outputL = bufferToFill.buffer->getWritePointer(0);
    float* const outputR = bufferToFill.buffer->getWritePointer(1);

    //Logger::getCurrentLogger()->outputDebugString("bufferSize: (" + juce::String(bufferToFill.numSamples) + ")");


    //AudioDeviceManager::AudioDeviceSetup currentAudioSetup;
    //deviceManager.getAudioDeviceSetup(currentAudioSetup);
    //Logger::getCurrentLogger()->outputDebugString("sample Rate in getNextAudioBlock: (" + juce::String(currentAudioSetup.sampleRate) + ")");
    //Logger::getCurrentLogger()->outputDebugString("buffer size in getNextAudioBlock: (" + juce::String(currentAudioSetup.bufferSize) + ")");



    for (int channel = 0; channel < bufferToFill.buffer->getNumChannels(); ++channel)
    {

        bp = x_inp_var;
        //bP = floor(bp * N); // same as bp = inp_bow_pos_x * L; bP = floor(bp / h);
        bP = floor(bp * N - 1); // -1 for alignment with Matlab..
        alpha_bow = bp * N - 1 - bP;

        int bP_m1 = bP - 1;
        int bP_p1 = bP + 1;
        int bP_p2 = bP + 2;

        std::fill(I_B.begin(), I_B.end(), 0); // fill with zeros 

        I_B[bP_m1] = (alpha_bow * (alpha_bow - 1) * (alpha_bow - 2)) / -6.0;
        I_B[bP] = ((alpha_bow - 1) * (alpha_bow + 1) * (alpha_bow - 2)) / 2.0;
        I_B[bP_p1] = (alpha_bow * (alpha_bow + 1) * (alpha_bow - 2)) / -2.0;
        I_B[bP_p2] = (alpha_bow * (alpha_bow + 1) * (alpha_bow - 1)) / 6.0;

        for (int i = 0; i < I_B.size(); ++i)
        {
            J_B[i] = I_B[i] * (1 / h); // speed up: keep divisions out of loop ! 
        }

        
        for (int i = 0; i < bufferToFill.numSamples; i++)
        {

            //Logger::getCurrentLogger()->outputDebugString("noSamples: (" + juce::String(bufferToFill.numSamples) + ")");
            //Logger::getCurrentLogger()->outputDebugString("i: (" + juce::String(i) + ")");

            if (channel == 0) //// this condition causes the output to explode sometimes
            {
                //c = 2 * f0 * L;
                //vB_var = sin(100.0 * 2.0 * float_Pi * t / fs) * 0.15 + 0.2;
                //vB_var = 0.2;
                //x_inp_var = sin(2.0 * float_Pi * 1000.0 * t)*0.1 + 0.4;

                //vB_var = vB;
                //fN_var = fN;
                
                //// For debugging:
                //if (t == 20)
                //{
                //    fN_var = 0.3;
                //    vB_var = 0.06;
                //    fS_var = fN_var * mu_S;
                //    fC_var = fN_var * mu_C;
                //    z_ba_var = 0.7 * (mu_C * fN_var) / s0;
                //    s3_var = s3_fac * fN;
                //}

                //// For debugging:
                //if (t == 30)
                //{
                //    f0 = 500;
                //    //fN_var = 0.3;
                //    //vB_var = 0.06;
                //    //fS_var = fN_var * mu_S;
                //    //fC_var = fN_var * mu_C;
                //    //z_ba_var = 0.7 * (mu_C * fN_var) / s0;
                //    //s3_var = s3_fac * fN;
                //}


                w_rnd = -1 + 2 * ((double)rand() / (RAND_MAX)); // random number between 0 and 1

                //Logger::getCurrentLogger()->outputDebugString("vel: (" + juce::String(vel) + ")");



                //I_S_J_S = sum(J_S(:).*J_S(:).*h);
                //I_P_J_P = sum(J_P(:).*J_P(:).*h_plate ^ 2);
                //% I_S_J_B = sum(J_S(:).*J_B(:).*h);
                //I_S_u_n = sum(J_S(:).*u(:).*h);
                //I_S_u_n_min_1 = sum(J_S(:).*uPrev(:).*h);
                //I_P_w_n = sum(J_P(:).*w(:).*h_plate ^ 2);
                //I_P_w_n_min_1 = sum(J_P(:).*wPrev(:).*h_plate ^ 2);

                //I_B_J_B = sum(J_B(:).*J_B(:).*h);
                //I_B_J_S = sum(J_B(:).*J_S(:).*h);
                //I_S_J_S = sum(J_S(:).*J_S(:).*h);
                //I_P_J_P = sum(J_P(:).*J_P(:).*h_plate ^ 2);
                //I_S_J_B = sum(J_S(:).*J_B(:).*h);
                //I_S_u_n = sum(J_S(:).*u(:).*h);
                //I_S_u_n_min_1 = sum(J_S(:).*uPrev(:).*h);
                //I_P_w_n = sum(J_P(:).*w(:).*h_plate ^ 2);
                //I_P_w_n_min_1 = sum(J_P(:).*wPrev(:).*h_plate ^ 2);


                double I_B_J_B = 0;
                double I_B_u = 0;
                double I_B_uPrev = 0;
                double I_B_dxx_u = 0;
                double I_B_dxx_uPrev = 0;
                double I_B_dxxxx_u = 0;
                double I_S_J_S = 0;
                double I_S_u = 0;
                double I_S_uPrev = 0;
                double I_S_dxx_u = 0;
                double I_S_dxx_uPrev = 0;
                double I_S_dxxxx_u = 0;
                double I_S_J_B = 0;
                int idx_p1;
                int idx_p2;
                int idx_m1;
                int idx_m2;

                for (int idx = 2; idx < N-2; ++idx)
                {
                    idx_p1 = idx + 1;
                    idx_p2 = idx + 2;
                    idx_m1 = idx - 1;
                    idx_m2 = idx - 2;

                    I_B_J_B = I_B_J_B + I_B[idx] * J_B[idx];
                    I_B_u = I_B_u + I_B[idx] * u[1][idx];
                    I_B_uPrev = I_B_uPrev + I_B[idx] * u[2][idx];

                    I_B_dxx_u = I_B_dxx_u + I_B[idx] * (u[1][idx_p1] - 2. * u[1][idx] + u[1][idx_m1]) * (1 / (h * h));
                    I_B_dxx_uPrev = I_B_dxx_uPrev + I_B[idx] * (u[2][idx_p1] - 2. * u[2][idx] + u[2][idx_m1]) * (1 / (h * h));

                    I_B_dxxxx_u = I_B_dxxxx_u
                        + I_B[idx] * (u[1][idx_p2] - 4. * u[1][idx_p1] + 6. * u[1][idx] - 4. * u[1][idx_m1] + u[1][idx_m2]) * (1 / (h * h * h * h));

                    I_S_J_S = I_S_J_S + I_S[idx] * J_S[idx];
                    //uVecs;
                    I_S_u = I_S_u + I_S[idx] * u[1][idx];
                    I_S_uPrev = I_S_uPrev + I_S[idx] * u[2][idx];

                    I_S_dxx_u = I_S_dxx_u + I_S[idx] * (u[1][idx_p1] - 2. * u[1][idx] + u[1][idx_m1]) * (1 / (h * h));
                    I_S_dxx_uPrev = I_S_dxx_uPrev + I_S[idx] * (u[2][idx_p1] - 2. * u[2][idx] + u[2][idx_m1]) * (1 / (h * h));

                    I_S_dxxxx_u = I_S_dxxxx_u
                        + I_S[idx] * (u[1][idx_p2] - 4. * u[1][idx_p1] + 6. * u[1][idx] - 4. * u[1][idx_m1] + u[1][idx_m2]) * (1 / (h * h * h * h));
                    I_S_J_B = I_S_J_B + I_S[idx] * J_B[idx];
                }

                double I_P_J_P = 0;
                double I_P_w = 0;
                double I_P_wPrev = 0;
                double I_P_dxx_w = 0;
                double I_P_dyy_w = 0;
                double I_P_delta_laplace_w = 0;
                double I_P_dxx_wPrev = 0;
                double I_P_dyy_wPrev = 0;
                double I_P_delta_laplace_wPrev = 0;

                double I_P_dxxxx_w = 0;
                double I_P_dyyyy_w = 0;
                double x2_I_P_dxxyy_w = 0;
                double I_P_delta_laplace_x2_w = 0;

                int idx_x;
                int idx_x_p1;
                int idx_x_p2;
                int idx_x_m1;
                int idx_x_m2;
                int idx_y;
                int idx_y_p1;
                int idx_y_p2;
                int idx_y_m1;
                int idx_y_m2;

                int idx_x_p1_y_p1;
                int idx_x_p1_y;
                int idx_x_p1_y_m1;
                int idx_x_y_p1;
                int idx_x_y;
                int idx_x_y_m1;
                int idx_x_m1_y_p1;
                int idx_x_m1_y;
                int idx_x_m1_y_m1;


                for (int iX = l_inp_plate; iX < l_inp_plate_plus1 + 1; ++iX) // for all the other points not at the connectin I_P is zero ! 
                {
                    for (int iY = m_inp_plate; iY < m_inp_plate_plus1 + 1; ++iY)
                    {
                        idx_x = iY + (iX)*N_plate;
                        idx_x_p1 = iY + (iX + 1) * N_plate;
                        idx_x_p2 = iY + (iX + 2) * N_plate;
                        idx_x_m1 = iY + (iX - 1) * N_plate;
                        idx_x_m2 = iY + (iX - 2) * N_plate;
                        idx_y = iY + (iX)*N_plate;
                        idx_y_p1 = (iY + 1) + (iX)*N_plate;
                        idx_y_p2 = (iY + 2) + (iX)*N_plate;
                        idx_y_m1 = (iY - 1) + (iX)*N_plate;
                        idx_y_m2 = (iY - 2) + (iX)*N_plate;

                        idx_x_p1_y_p1 = (iY + 1) + (iX + 1) * N_plate;
                        idx_x_p1_y = (iY)+(iX + 1) * N_plate;
                        idx_x_p1_y_m1 = (iY - 1) + (iX + 1) * N_plate;
                        idx_x_y_p1 = (iY + 1) + (iX) * N_plate;
                        idx_x_y = (iY)+(iX)*N_plate;
                        idx_x_y_m1 = (iY - 1) + (iX)*N_plate;
                        idx_x_m1_y_p1 = (iY + 1) + (iX - 1) * N_plate;
                        idx_x_m1_y = (iY)+(iX - 1) * N_plate;
                        idx_x_m1_y_m1 = (iY-1)+(iX - 1) * N_plate;

                        I_P_J_P = I_P_J_P + I_P[idx_x] * J_P[idx_x];

                        I_P_w = I_P_w + w[1][idx_x] * I_P[idx_x];
                        I_P_wPrev = I_P_wPrev + w[2][idx_x] * I_P[idx_x];

                        I_P_dxx_w = I_P_dxx_w + I_P[idx_x] * (w[1][idx_x_p1] - 2. * w[1][idx_x] + w[1][idx_x_m1]) * (1 / (h_plate * h_plate));
                        I_P_dyy_w = I_P_dyy_w + I_P[idx_y] * (w[1][idx_y_p1] - 2. * w[1][idx_y] + w[1][idx_y_m1]) * (1 / (h_plate * h_plate));
                        I_P_delta_laplace_w = I_P_dxx_w + I_P_dxx_w;

                        I_P_dxx_wPrev = I_P_dxx_wPrev + I_P[idx_x] * (w[2][idx_x_p1] - 2. * w[2][idx_x] + w[2][idx_x_m1]) * (1 / (h_plate * h_plate));
                        I_P_dyy_wPrev = I_P_dyy_wPrev + I_P[idx_y] * (w[2][idx_y_p1] - 2. * w[2][idx_y] + w[2][idx_y_m1]) * (1 / (h_plate * h_plate));
                        I_P_delta_laplace_wPrev = I_P_dxx_wPrev + I_P_dyy_wPrev;

                        I_P_dxxxx_w = I_P_dxxxx_w + I_P[idx_x] * (w[1][idx_x_p2] - 4. * w[1][idx_x_p1] + 6. * w[1][idx_x] - 4. * w[1][idx_x_m1] + w[1][idx_x_m2]) * (1 / (h_plate * h_plate * h_plate * h_plate));
                        I_P_dyyyy_w = I_P_dyyyy_w + I_P[idx_y] * (w[1][idx_y_p2] - 4. * w[1][idx_y_p1] + 6. * w[1][idx_y] - 4. * w[1][idx_y_m1] + w[1][idx_y_m2]) * (1 / (h_plate * h_plate * h_plate * h_plate));
                        x2_I_P_dxxyy_w = x2_I_P_dxxyy_w + I_P[idx_x]
                            * (w[1][idx_x_p1_y_p1] - 2. * w[1][idx_x_p1_y] + w[1][idx_x_p1_y_m1] - 2. * w[1][idx_x_y_p1] + 4. * w[1][idx_x_y]
                                - 2. * w[1][idx_x_y_m1] + w[1][idx_x_m1_y_p1] - 2. * w[1][idx_x_m1_y] + w[1][idx_x_m1_y_m1]) * (2 / (h_plate * h_plate * h_plate * h_plate));
                        
                        I_P_delta_laplace_x2_w = I_P_dxxxx_w + I_P_dyyyy_w + x2_I_P_dxxyy_w;
                    }
                }


                double q = (-2 / k) * (1 / k) * (I_B_u - I_B_uPrev)
                            + (2 / k) * vB_var
                            + 2 * sig0 * vB_var
                            - (c * c) * I_B_dxx_u 
                            + K * K * I_B_dxxxx_u
                            - (2 * sig1) * (1 / k) * I_B_dxx_u
                            + (2 * sig1) * (1 / k) * I_B_dxx_uPrev;

                double b = (k * k / (1 + sig0 * k)) * (c * c  * I_S_dxx_u - K * K * I_S_dxxxx_u + (2 * sig1 / k) * (I_S_dxx_u - I_S_dxx_uPrev))
                         + (k * k / (1 + sig0 * k)) * (2 / (k * k)) * I_S_u - (k * k / (1 + sig0 * k)) * ((1 - sig0 * k) / (k * k)) * I_S_uPrev
                         - (k * k / (1 + sig0_plate * k)) * (-(K_plate * K_plate) * I_P_delta_laplace_x2_w + (2 * sig1_plate / k) * (I_P_delta_laplace_w - I_P_delta_laplace_wPrev))
                         - (k * k / (1 + sig0_plate * k)) * (2 / (k * k)) *I_P_w + (k * k / (1 + sig0_plate * k)) * ((1 - sig0_plate * k) / (k * k)) * I_P_wPrev;


                    double eps = 1;
                    //w_rnd_last = -1 + (1 - (-1)).*rand(1);

                    int iter_check = 0;
                    vrel_last = -vB;
                    z_last = 0;
                    // Newton-Raphson iterative scheme
                    while ((eps > tol) && (fC_var > 0))
                    {
                        ++iter_check;

                        if (vrel_last == 0)
                        {
                            z_ss = fS_var / s0;
                        }
                        else
                        {
                            z_ss = sign(vrel_last) / s0 * (fC_var + (fS_var - fC_var) * exp(-pow((vrel_last / vS), 2)));
                        }

                        if (sign(vrel_last) == sign(z_last))
                        {
                            if (abs(z_last) <= z_ba_var)
                            {
                                alpha_fct = 0;
                            }
                            else if ((z_ba_var < abs(z_last)) && (abs(z_last) < abs(z_ss)))
                            {
                                alpha_fct = 0.5 * (1 + sign(z_last) * sin((double_Pi * (z_last - sign(z_last) * 0.5 * (abs(z_ss) + z_ba_var)) / (abs(z_ss) - z_ba_var))));
                            }
                            else if (abs(z_last) >= abs(z_ss))
                            {
                                alpha_fct = 1;
                            }
                        }
                        else
                        {
                            alpha_fct = 0;
                        }


                        r_last = vrel_last * (1 - alpha_fct * z_last / z_ss);
                        a_last = ((2 / k) * (z_last - z_prev) - a_prev);

                        g1 = I_B_J_B * ((s0 * z_last + s1 * r_last + s2 * vrel_last + s3_var * w_rnd) / (rho * A)) + (2 / k + 2 * sig0) * vrel_last + q;
                        g2 = r_last - ((2 / k) * (z_last - z_prev) - a_prev);

                        if (sign(vrel_last) >= 0)
                        {
                            z_ss_d_vrel = -2 * vrel_last * (-fC_var + fS_var) * exp(-(vrel_last * vrel_last) / (vS * vS)) / (s0 * (vS * vS));
                        }
                        else
                        {
                            z_ss_d_vrel = 2 * vrel_last * (-fC_var + fS_var) * exp(-(vrel_last * vrel_last) / (vS * vS)) / (s0 * (vS * vS));
                        }

                        if ((z_ba_var < abs(z_last)) && (abs(z_last) < abs(z_ss)))
                        {
                            if (sign(z_last) >= 0)
                            {
                                alpha_fct_d_vrel = 0.5 * (-0.5 * double_Pi * (z_ss * z_ss_d_vrel) * sign(z_ss) / ((-z_ba_var + abs(z_ss)) * z_ss) - double_Pi * (z_ss * z_ss_d_vrel) * (-0.5 * z_ba_var + z_last - 0.5 * abs(z_ss)) * sign(z_ss) / (pow((-z_ba_var + abs(z_ss)), 2) * z_ss)) * cos(double_Pi * (-0.5 * z_ba_var + z_last - 0.5 * abs(z_ss)) / (-z_ba_var + abs(z_ss)));
                                alpha_fct_d_z = 0.5 * double_Pi * cos(double_Pi * (-0.5 * z_ba_var + z_last - 0.5 * abs(z_ss)) / (-z_ba_var + abs(z_ss))) / (-z_ba_var + abs(z_ss));
                            }
                            else
                            {
                                alpha_fct_d_vrel = -0.5 * (0.5 * double_Pi * (z_ss * z_ss_d_vrel) * sign(z_ss) / ((-z_ba_var + abs(z_ss)) * z_ss) - double_Pi * (z_ss * z_ss_d_vrel) * (0.5 * z_ba_var + z_last + 0.5 * abs(z_ss)) * sign(z_ss) / (pow((-z_ba_var + abs(z_ss)), 2) * z_ss)) * cos(double_Pi * (0.5 * z_ba_var + z_last + 0.5 * abs(z_ss)) / (-z_ba_var + abs(z_ss)));
                                alpha_fct_d_z = -0.5 * double_Pi * cos(double_Pi * (0.5 * z_ba_var + z_last + 0.5 * abs(z_ss)) / (-z_ba_var + abs(z_ss))) / (-z_ba_var + abs(z_ss));
                            }
                        }
                        else
                        {
                            alpha_fct_d_vrel = 0;
                            alpha_fct_d_z = 0;
                        }

                        r_last_d_vrel = vrel_last * (z_last * alpha_fct * z_ss_d_vrel / (z_ss * z_ss) - z_last * alpha_fct_d_vrel / z_ss) - z_last * alpha_fct / z_ss + 1;
                        r_last_d_z = vrel_last * (-z_last * alpha_fct_d_z / z_ss - alpha_fct / z_ss);

                        g1_d_vrel = 2 * sig0 + 2 / k + I_B_J_B * (s1 * r_last_d_vrel + s2) / (rho * A);
                        g1_d_z = I_B_J_B * (s0 + s1 * r_last_d_z) / (rho * A);

                        g2_d_vrel = r_last_d_vrel;
                        g2_d_z = r_last_d_z - 2 / k;

                        fac_1_over_det = 1 / (g1_d_vrel * g2_d_z - g1_d_z * g2_d_vrel);

                        A_mat_new = fac_1_over_det * g2_d_z;
                        B_mat_new = fac_1_over_det * (-g1_d_z);
                        C_mat_new = fac_1_over_det * (-g2_d_vrel);
                        D_mat_new = fac_1_over_det * (g1_d_vrel);

                        vrel = vrel_last - (A_mat_new * g1 + B_mat_new * g2);
                        z = z_last - (C_mat_new * g1 + D_mat_new * g2);

                        eps = sqrt((z - z_last) * (z - z_last) + (vrel - vrel_last) * (vrel - vrel_last)); // same as norm(theta - theta_last);

                        z_last = z;
                        vrel_last = vrel;

                        if (iter_check == 99)
                            break;
                    }

                    if (fC_var > 0)
                    {
                        if (vrel == 0)
                        {
                            z_ss = fS_var / s0;
                        }
                        else
                        {
                            z_ss = sign(vrel) / s0 * (fC_var + (fS_var - fC_var) * exp(-(vrel / vS) * (vrel / vS)));
                        }

                        double ana = sign(vrel);

                        if (sign(vrel) == sign(z))
                        {
                            if (abs(z) <= z_ba_var)
                            {
                                alpha_fct = 0;
                            }
                            else if ((z_ba_var < abs(z)) && (abs(z) < abs(z_ss)))
                            {
                                alpha_fct = 0.5 * (1 + sign(z) * sin((double_Pi * (z - sign(z) * 0.5 * (abs(z_ss) + z_ba_var)) / (abs(z_ss) - z_ba_var))));
                            }
                            else if (abs(z) >= abs(z_ss))
                            {
                                alpha_fct = 1;
                            }
                        }
                        else
                        {
                            alpha_fct = 0;
                        }

                        r = vrel * (1 - alpha_fct * z / z_ss);
                        a = (2 / k) * (z - z_prev) - a_prev;


                        Fbow = (s0 * z + s1 * r + s2 * vrel + s3_var * w_rnd);


                        z_prev = z;
                        a_prev = a;
                    }
                    else
                    {
                        Fbow = 0;

                        // reset params for NR
                        z_prev = 0;
                        r_last = vB_var;
                        r_prev = vB_var;
                        a_last = r_last;
                        a_prev = r_prev;

                        vrel_last = -vB_var;
                        z_last = 0;

                        F_last = 0;
                    }

                F = (-b + (k * k / (1 + sig0 * k)) * I_S_J_B * ((s0 * z_last + s1 * r_last + s2 * vrel_last + s3 * w_rnd) / (rho * A))) / (k * k * I_S_J_S / ((1 + sig0 * k) * rho * A) + k * k * I_P_J_P / ((1 + sig0_plate * k) * rho_plate * H_plate));

                // Update equations
                for (int idx = 2; idx < N - 2; ++idx)
                {
                    idx_p1 = idx + 1;
                    idx_p2 = idx + 2;
                    idx_m1 = idx - 1;
                    idx_m2 = idx - 2;

                     // maybe you can reused the definitions from above I_S_etc

                    u[0][idx] = (k * k / (1 + sig0 * k)) * ((c * c / (h * h)) * (u[1][idx_p1] - 2 * u[1][idx] + u[1][idx_m1])
                            + (2 * sig1 / (k * h * h)) * (u[1][idx_p1] - 2 * u[1][idx] + u[1][idx_m1] - u[2][idx_p1] + 2 * u[2][idx] - u[2][idx_m1])
                            - (K * K) * (1 / (h * h * h * h)) * (u[1][idx_p2] - 4 * u[1][idx_p1] + 6 * u[1][idx] - 4 * u[1][idx_m1] + u[1][idx_m2])
                            - J_B[idx] * Fbow / (rho * A) + J_S[idx] * F / (rho * A)
                            + (2 / (k * k)) * u[1][idx] - (1 - sig0 * k) * u[2][idx] / (k * k));
                }

                //// Add simply supperted BCs
                //int idx = 1;
                //idx_p1 = idx + 1;
                //idx_p2 = idx + 2;
                //idx_m1 = idx - 1;
                //idx_m2 = idx - 2;

                //u[0][idx] = (k * k / (1 + sig0 * k)) * ((c * c / (h * h)) * (u[1][idx_p1] - 2 * u[1][idx] + u[1][idx_m1])
                //    + (2 * sig1 / (k * h * h)) * (u[1][idx_p1] - 2 * u[1][idx] + u[1][idx_m1] - u[2][idx_p1] + 2 * u[2][idx] - u[2][idx_m1])
                //    - (K * K) * (1 / (h * h * h * h)) * (u[1][idx_p2] - 4 * u[1][idx_p1] + 6 * u[1][idx] - 4 * u[1][idx_m1] - u[1][idx])
                //    - J_B[idx] * Fbow / (rho * A) + J_S[idx] * F / (rho * A)
                //    + (2 / (k * k)) * u[1][idx] - (1 - sig0 * k) * u[2][idx] / (k * k));

                u[0][1] = (k * k / (1 + sig0 * k)) * ((c * c / (h * h)) * (u[1][2] - 2 * u[1][1] + u[1][0])
                    + (2 * sig1 / (k * h * h)) * (u[1][2] - 2 * u[1][1] + u[1][0] - u[2][2] + 2 * u[2][1] - u[2][0])
                    - (K * K) * (1 / (h * h * h * h)) * (u[1][3] - 4 * u[1][2] + 6 * u[1][1] - 4 * u[1][0] - u[1][1])
                    - J_B[1] * Fbow / (rho * A) + J_S[1] * F / (rho * A)
                    + (2 / (k * k)) * u[1][1] - (1 - sig0 * k) * u[2][1] / (k * k));

                int idx = N - 2;
                idx_p1 = idx + 1;
                idx_p2 = idx + 2;
                idx_m1 = idx - 1;
                idx_m2 = idx - 2;

                u[0][idx] = (k * k / (1 + sig0 * k)) * ((c * c / (h * h)) * (u[1][idx_p1] - 2 * u[1][idx] + u[1][idx_m1])
                    + (2 * sig1 / (k * h * h)) * (u[1][idx_p1] - 2 * u[1][idx] + u[1][idx_m1] - u[2][idx_p1] + 2 * u[2][idx] - u[2][idx_m1])
                    - (K * K) * (1 / (h * h * h * h)) * (-u[1][idx] - 4 * u[1][idx_p1] + 6 * u[1][idx] - 4 * u[1][idx_m1] + u[1][idx_m2])
                    - J_B[idx] * Fbow / (rho * A) + J_S[idx] * F / (rho * A)
                    + (2 / (k * k)) * u[1][idx] - (1 - sig0 * k) * u[2][idx] / (k * k));


                for (int iLoc = 0; iLoc < locsDo_X.size(); ++iLoc)
                {
                    int iX = locsDo_X[iLoc];
                    int iY = locsDo_Y[iLoc];

                    idx_x = iY + (iX)*N_plate;
                    idx_x_p1 = iY + (iX + 1) * N_plate;
                    idx_x_p2 = iY + (iX + 2) * N_plate;
                    idx_x_m1 = iY + (iX - 1) * N_plate;
                    idx_x_m2 = iY + (iX - 2) * N_plate;
                    idx_y = iY + (iX)*N_plate;
                    idx_y_p1 = (iY + 1) + (iX)*N_plate;
                    idx_y_p2 = (iY + 2) + (iX)*N_plate;
                    idx_y_m1 = (iY - 1) + (iX)*N_plate;
                    idx_y_m2 = (iY - 2) + (iX)*N_plate;

                    idx_x_p1_y_p1 = (iY + 1) + (iX + 1) * N_plate;
                    idx_x_p1_y = (iY)+(iX + 1) * N_plate;
                    idx_x_p1_y_m1 = (iY - 1) + (iX + 1) * N_plate;
                    idx_x_y_p1 = (iY + 1) + (iX)*N_plate;
                    idx_x_y = (iY)+(iX)*N_plate;
                    idx_x_y_m1 = (iY - 1) + (iX)*N_plate;
                    idx_x_m1_y_p1 = (iY + 1) + (iX - 1) * N_plate;
                    idx_x_m1_y = (iY)+(iX - 1) * N_plate;
                    idx_x_m1_y_m1 = (iY - 1) + (iX - 1) * N_plate;


                    double fac1 = (w[1][idx_x_p2] - 4 * w[1][idx_x_p1] + 6 * w[1][idx_x] - 4 * w[1][idx_x_m1] + w[1][idx_x_m2]
                        + w[1][idx_y_p2] - 4 * w[1][idx_y_p1] + 6 * w[1][idx_y] - 4 * w[1][idx_y_m1] + w[1][idx_y_m2]
                        + 2 * (w[1][idx_x_p1_y_p1] - 2. * w[1][idx_x_p1_y] + w[1][idx_x_p1_y_m1] - 2 * w[1][idx_x_y_p1] + 4 * w[1][idx_x_y] - 2 * w[1][idx_x_y_m1] + w[1][idx_x_m1_y_p1] - 2 * w[1][idx_x_m1_y] + w[1][idx_x_m1_y_m1]));

                    w[0][idx_x_y] = (k * k / (1 + sig0_plate * k)) * ((-(K_plate * K_plate) / (h_plate * h_plate * h_plate * h_plate)) * fac1
                        + (2 * sig1_plate / k) * (1 / (h_plate * h_plate)) * (w[1][idx_x_p1] + w[1][idx_x_m1] + w[1][idx_y_p1] + w[1][idx_y_m1] - 4 * w[1][idx_x_y] - (w[2][idx_x_p1] + w[2][idx_x_m1] + w[2][idx_y_p1] + w[2][idx_y_m1] - 4 * w[2][idx_x_y]))
                        - J_P[idx_x_y] * F / (rho_plate * H_plate)
                        + (2 / (k * k)) * w[1][idx_x_y] - (1 - sig0_plate * k) * w[2][idx_x_y] / (k * k));

                }

                //for (int iX = 2; iX < N_plate - 2; ++iX)
                //{
                //    for (int iY = 2; iY < N_plate - 2; ++iY)
                //    {
                //        idx_x = iY + (iX)*N_plate;
                //        idx_x_p1 = iY + (iX + 1) * N_plate;
                //        idx_x_p2 = iY + (iX + 2) * N_plate;
                //        idx_x_m1 = iY + (iX - 1) * N_plate;
                //        idx_x_m2 = iY + (iX - 2) * N_plate;
                //        idx_y = iY + (iX)*N_plate;
                //        idx_y_p1 = (iY + 1) + (iX)*N_plate;
                //        idx_y_p2 = (iY + 2) + (iX)*N_plate;
                //        idx_y_m1 = (iY - 1) + (iX)*N_plate;
                //        idx_y_m2 = (iY - 2) + (iX)*N_plate;

                //        idx_x_p1_y_p1 = (iY + 1) + (iX + 1) * N_plate;
                //        idx_x_p1_y = (iY)+(iX + 1) * N_plate;
                //        idx_x_p1_y_m1 = (iY - 1) + (iX + 1) * N_plate;
                //        idx_x_y_p1 = (iY + 1) + (iX)*N_plate;
                //        idx_x_y = (iY)+(iX)*N_plate;
                //        idx_x_y_m1 = (iY - 1) + (iX)*N_plate;
                //        idx_x_m1_y_p1 = (iY + 1) + (iX - 1) * N_plate;
                //        idx_x_m1_y = (iY)+(iX - 1) * N_plate;
                //        idx_x_m1_y_m1 = (iY - 1) + (iX - 1) * N_plate;


                //        double fac1 = (w[1][idx_x_p2] - 4 * w[1][idx_x_p1] + 6 * w[1][idx_x] - 4 * w[1][idx_x_m1] + w[1][idx_x_m2]
                //              + w[1][idx_y_p2] - 4 * w[1][idx_y_p1] + 6 * w[1][idx_y] - 4 * w[1][idx_y_m1] + w[1][idx_y_m2]
                //              + 2 * (w[1][idx_x_p1_y_p1] - 2. * w[1][idx_x_p1_y] + w[1][idx_x_p1_y_m1] - 2 * w[1][idx_x_y_p1] + 4 * w[1][idx_x_y] - 2 * w[1][idx_x_y_m1] + w[1][idx_x_m1_y_p1] - 2 * w[1][idx_x_m1_y] + w[1][idx_x_m1_y_m1]));

                //        w[0][idx_x_y] = (k * k / (1 + sig0_plate * k)) * ((-(K_plate * K_plate) / (h_plate * h_plate * h_plate * h_plate)) * fac1
                //            + (2 * sig1_plate / k) * (1 / (h_plate * h_plate)) * (w[1][idx_x_p1] + w[1][idx_x_m1] + w[1][idx_y_p1] + w[1][idx_y_m1] - 4 * w[1][idx_x_y] - (w[2][idx_x_p1] + w[2][idx_x_m1] + w[2][idx_y_p1] + w[2][idx_y_m1] - 4 * w[2][idx_x_y]))
                //            - J_P[idx_x_y] * F / (rho_plate * H_plate)
                //            + (2 / (k * k)) * w[1][idx_x_y] - (1 - sig0_plate * k) * w[2][idx_x_y] / (k * k));


                //    }
                //}



                //              if (i == 10)
                //              {
                //                  //Logger::getCurrentLogger()->outputDebugString("J_grid: (" + String(J_grid[l_inp]) + ")");
                //                  //Logger::getCurrentLogger()->outputDebugString("J_grid: (" + String(J_grid[l_inp_plus1]) + ")");
                //                  //Logger::getCurrentLogger()->outputDebugString("uNext: (" + String(uNext[l_inp_plus1]) + ")");
                //              }

                     /*         int idx = floor(3 * N / 8);
                              preOutput = gain * u[0][62];*/

                              //Logger::getCurrentLogger()->outputDebugString("preOutput: (" + String(preOutput) + ")");




                //              //uPrev = u; // this copies an entire state vector point by point ( value by value ) this is expensive.
                //              //u = uNext; 

                double I_S_uNext = 0;


                for (int idx = 2; idx < N - 2; ++idx)
                {
                    I_S_uNext = I_S_uNext + I_S[idx] * u[0][idx];
                }

                //uVecs;

                preOutput = I_S_uNext;
                preOutput = preOutput * gain;


                //int l_out = floor(x_out_var / h);
                //int l_out_plus1 = l_out + 1;
                //int m_out = floor(y_out_var / h);
                //int m_out_plus1 = m_inp + 1;
                //double alpha_x_out = x_out_var / h - l_out;
                //double alpha_y_out = y_out_var / h - m_out;


                //preOutput = u[0][l_out * Nx + m_out] * (1 - alpha_x_out) * (1 - alpha_y_out) +
                //    u[0][(l_out + 1) * Nx + m_out] * alpha_x_out * (1 - alpha_y_out) +
                //    u[0][(l_out)*Nx + (m_out + 1)] * (1 - alpha_x_out) * alpha_y_out +
                //    u[0][(l_out + 1) * Nx + (m_out + 1)] * alpha_x_out * alpha_y_out;
                //preOutput = preOutput * gain;


                //// LOGGING

                //FileLogger(File("C:\\Marius\\UNIVERSITY\\SEMESTER_4\\MSc_Thesis_Project\\juce_physical_models\\\Funky_Violin_Rev2\\Logs\\test_Fbow.txt"), String(Fbow), 128000 * 1024);
                //FileLogger(File("C:\\Marius\\UNIVERSITY\\SEMESTER_4\\MSc_Thesis_Project\\juce_physical_models\\\Funky_Violin_Rev2\\Logs\\test_Fconn.txt"), String(F), 128000 * 1024);
                //FileLogger(File("C:\\Marius\\UNIVERSITY\\SEMESTER_4\\MSc_Thesis_Project\\juce_physical_models\\\Funky_Violin_Rev2\\Logs\\test_vrel.txt"), String(vrel), 128000 * 1024);
                //FileLogger(File("C:\\Marius\\UNIVERSITY\\SEMESTER_4\\MSc_Thesis_Project\\juce_physical_models\\\Funky_Violin_Rev2\\Logs\\output_test.txt"), String(I_S_uNext), 128000 * 1024);
                //FileLogger(File("C:\\Marius\\UNIVERSITY\\SEMESTER_4\\MSc_Thesis_Project\\juce_physical_models\\\Funky_Violin_Rev2\\Logs\\q_test.txt"), String(q), 128000 * 1024);
                //FileLogger(File("C:\\Marius\\UNIVERSITY\\SEMESTER_4\\MSc_Thesis_Project\\juce_physical_models\\\Funky_Violin_Rev2\\Logs\\b_test.txt"), String(b), 128000 * 1024);

                //if (i == 0)
                //{
                //    Logger::getCurrentLogger()->outputDebugString("fN_var: (" + String(fN_var) + ")");
                //    Logger::getCurrentLogger()->outputDebugString("fC_var: (" + String(fC_var) + ")");
                //    Logger::getCurrentLogger()->outputDebugString("vB_var: (" + String(vB_var) + ")");
                //    Logger::getCurrentLogger()->outputDebugString("x_inp_var: (" + String(x_inp_var) + ")");
                //    Logger::getCurrentLogger()->outputDebugString("preOutput: (" + String(preOutput) + ")");
                //    Logger::getCurrentLogger()->outputDebugString("c: (" + String(c) + ")");
                //    Logger::getCurrentLogger()->outputDebugString("y_inp_var: (" + String(y_inp_var) + ")");
                //}






                //Logger::getCurrentLogger()->outputDebugString("t: (" + String(t) + ")");
                //Logger::getCurrentLogger()->outputDebugString("Fbow: (" + String(Fbow) + ")");
                //Logger::getCurrentLogger()->outputDebugString("F: (" + String(F) + ")");
                //Logger::getCurrentLogger()->outputDebugString("vrel: (" + String(vrel) + ")");
                //Logger::getCurrentLogger()->outputDebugString("z: (" + String(z) + ")");
                //Logger::getCurrentLogger()->outputDebugString("I_S_uNext: (" + String(I_S_uNext) + ")");
                //Logger::getCurrentLogger()->outputDebugString("q: (" + String(q) + ")");
                //Logger::getCurrentLogger()->outputDebugString("b: (" + String(b) + ")");



  //              //double* ptr; 
  //              //if (I_grid.size() != 0)
  //              //{
  //              //    ptr = &I_grid[0];
  //              //}


                uTmp = u[2];
                u[2] = u[1];
                u[1] = u[0];
                u[0] = uTmp;

                wTmp = w[2];
                w[2] = w[1];
                w[1] = w[0];
                w[0] = wTmp;

                //if (i == 0)
                //{
                //    Logger::getCurrentLogger()->outputDebugString("preOutput: (" + String(preOutput) + ")");
                //}

                //preOutput = gain * sineWave;
                if (abs(preOutput) > 1)
                {
                    /*std::cout << "Output is too loud!" << std::endl;*/
                    //Logger::getCurrentLogger()->outputDebugString("Output is too loud!");
                }
                else {
                    //output_interim[i] = preOutput;
                    outputL[i] = preOutput;
                    ////outputR[i] = gain * uNext[floor(6 * N / 8)];
                    outputR[i] = outputL[i];
                }

                //uVecs

                //Logger::getCurrentLogger()->outputDebugString(String(t));
                ++t;
                int anabanana = 5;

            }

        }
    }



}

void MainComponent::releaseResources()
{
    // This will be called when the audio device stops, or when it is being
    // restarted due to a setting change.

    // For more details, see the help for AudioProcessor::releaseResources()
}

//==============================================================================
//==============================================================================
void MainComponent::paint(juce::Graphics& g)
{
    //// (Our component is opaque, so we must completely fill the background with a solid colour)
    //g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

    // You can add your drawing code here!

    float stateWidth = getWidth() / static_cast<double> (N_plate);
    float stateHeight = getHeight() / static_cast<double> (N_plate);
    //int scaling = 0.1 / (accumulate(scaleAdjustVec.begin(), scaleAdjustVec.end(), 0.0) / movavg_size); 

    int scaling = 10000;


    //for (int x = 0; x < N_plate; ++x)
    //{
    //    for (int y = 0; y < N_plate; ++y)
    //    {
    //        int idx_x_y = y + (x)*N_plate;

    //        int cVal = clamp(255 * 0.5 * (w[1][idx_x_y] * scaling + 1), 0, 255);

    //        //g.setColour(Colour::fromRGBA(cVal, cVal, cVal, 127));
    //        g.setColour(Colours::orange);

    //        //g.setColour(Colour::fromRGBA(cVal, cVal, cVal, 127));
    //        g.fillRect((x)*stateWidth, (y)*stateHeight, stateWidth, stateHeight);
    //    }
    //}

    for (int iLoc = 0; iLoc < locsDo_X.size(); ++iLoc)
    {
        int x = locsDo_X[iLoc];
        int y = locsDo_Y[iLoc];

        int idx_x_y = y + (x)*N_plate;

        int cVal = clamp(255 * 0.5 * (w[1][idx_x_y] * scaling + 1), 0, 255);

        g.setColour(Colour::fromRGBA(cVal, cVal, cVal, 127));
        //g.setColour(Colours::orange);

        //g.setColour(Colour::fromRGBA(cVal, cVal, cVal, 127));
        g.fillRect((x)*stateWidth, (y)*stateHeight, stateWidth, stateHeight);
    }

}

void MainComponent::resized()
{
    // This is called when the MainContentComponent is resized.
    // If you add any child components, this is where you should
    // update their positions.

    auto sliderLeft = 120;
    frequencySlider.setBounds(sliderLeft, 20, getWidth() - sliderLeft - 10, 20);

    sig_0_Slider.setBounds(sliderLeft, 50, getWidth() - sliderLeft - 10, 20);

    sig_1_Slider.setBounds(sliderLeft, 80, getWidth() - sliderLeft - 10, 20);

}




void MainComponent::timerCallback()
{
    if (graphicsToggle)
        repaint();
    //    std::cout << instruments[0]->getStrings()[0]->isStringBowing() << std::endl;
}


void MainComponent::hiResTimerCallback()
{
    for (auto sensel : sensels)
    {
        double finger0X = 0;
        double finger0Y = 0;
        if (sensel->senselDetected)
        {
            sensel->check();
            unsigned int fingerCount = sensel->contactAmount;
            int index = sensel->senselIndex;
            //if (!easyControl)
            //    trombaString->setFingerForce(0.0);
            for (int f = 0; f < fingerCount; f++)
            {
                bool state = sensel->fingers[f].state;
                float x = sensel->fingers[f].x;
                float y = sensel->fingers[f].y;

                x_inp_var = clamp(x, 0.2, 0.8);
                f0 = linearMapping(1.0, 0.0, 440.0, 100.0, y);
                c = 2 * f0 * L;

                //vB_var = clamp(y, 0.2, 0.9);


                float Vb = -sensel->fingers[f].delta_y * 0.5; // this is interesting ! 

                //float fff = clamp(sensel->fingers[f].force, 0, fN);
                //fN_var = linearMapping(0.2, 0, fN, 0.1, fff);

                //vB_var = linearMapping(0, fN, 0, vB, fN_var);

                //float fff = clamp(sensel->fingers[f].force, 0.01, fN / 3);
                ////fN_var = linearMapping(fN / 3, 0.01, fN, 0.01, fff);

                fN_var = sensel->fingers[f].force;
                fN_var = linearMapping(0.1, 0.0, fN, 0.0, fN_var);
                fN_var = clamp(fN_var, 0.01, fN);

                vB_var = linearMapping(fN, 0.01, vB, 0.001, fN_var);

                //fN_var = y * fN;
                //vB_var = y * vB;

                //fN_var = 7;
                //vB_var = 0.1;

                //Logger::getCurrentLogger()->outputDebugString("vB_var: (" + String(vB_var) + ")");
                //Logger::getCurrentLogger()->outputDebugString("fN_var: (" + String(fN_var) + ")");

                //int idx_x_y = locsDo[5][1] + locsDo[5][0] * N_membrane;
                //double uPrint = u[1][idx_x_y];

                //Logger::getCurrentLogger()->outputDebugString("uPrint: (" + String(uPrint) + ")");


                //vB_var = clamp(sensel->fingers[f].force, 0, vB);
                //float Fn = clamp(sensel->fingers[f].force * 5.0, 0, fN);


                fS_var = fN_var * mu_S;
                fC_var = fN_var * mu_C;
                z_ba_var = 0.7 * (mu_C * fN_var) / s0;
                s3_var = s3_fac * fN_var;


                //int fingerID = sensel->fingers[f].fingerID;

            }

            if (fingerCount == 0)
            {
                // this can just be a reset_bow() function
                fN_var = 0.0; // if fN is 0, don't do the NR ! just add condition in the while of the NR
                vB_var = 0.0;
                F = 0.0;
                fS_var = fN_var * mu_S;
                fC_var = fN_var * mu_C;
                z_ba_var = 0.7 * (mu_C * fN_var) / s0;
                s3_var = s3_fac * fN_var;
            }
        }
    }
}














