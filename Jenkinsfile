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
                   ./gradlew build
                   ./gradlew setupVenv
                '''
            }
        }

        stage('Start Streamlit Application') {
            steps {
                sh '''
                   ./gradlew runDemo
                '''
            }
        }
    }
}
