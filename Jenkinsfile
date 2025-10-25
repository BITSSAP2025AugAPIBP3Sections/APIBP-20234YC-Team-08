pipeline {
    // This tells Jenkins to run the job on any available agent
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                // Clones your project from the repository configured in the Jenkins job
                echo 'Cloning the repository...'
                checkout scm
            }
        }

        stage('Setup Python Environment') {
            steps {
                // This block runs multiple shell commands in one go
                sh '''
                    echo "Setting up Python virtual environment..."
                    # Remove old venv if it exists
                    rm -rf venv
                    
                    # Create and activate the virtual environment
                    # python3 -m venv venv
                    source linux_venv/bin/activate
                    
                    echo "Installing dependencies..."
                   # pip3 install -r requirements.txt
                '''
            }
        }

        stage('Start Streamlit Application') {
            steps {
                sh '''
                    echo "Stopping any old Streamlit process..."
                    # This command finds and stops any running streamlit process to avoid port conflicts.
                    # '|| true' ensures the build doesn't fail if no process is found.
                    pkill -f streamlit || true

                    echo "Starting Streamlit application in the background..."
                    # Activate the virtual environment
                    source linux_venv/bin/activate
                

                
                    # 'nohup' runs the command even if you close the terminal.
                    # '&' runs the command in the background, so it doesn't block the Jenkins build.
                    nohup python -m streamlit run src/app.py --server.port 8501 &
                '''
            }
        }
    }
}
