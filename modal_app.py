import modal
import subprocess
import os

# 1. Replicating your Dockerfile
# We match Python 3.12 and install the specific system libraries you listed.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")  # From your Dockerfile RUN command
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root/app", ignore=[
        ".venv",        # Exclude virtual env
            "__pycache__",  # Exclude bytecode cache
            ".git",         # Exclude git repo
            "*.pyc",        # Exclude loose bytecode files
            "*.pyo",        # Exclude optimized bytecode
            "*.log",        # Exclude logs if any
            ".env",
            "simple_analysis.py",
            "secrets/",
            "volumes/",
            "data/",
            ".vscode/",
            "*.json",
            "RAGAS_test_details/",          
            "zilliz-cloud-Dissertation-username-password.txt",
        ],) # Replaces COPY . .
)

app = modal.App("neatstandards-rag")

# 2. Define Secrets (Environment Variables)
# You must create a secret named "neatstandards-secrets" in the Modal dashboard
# containing: NEBIUS_API_KEY, OPENAI_API_KEY, CHAINLIT_AUTH_SECRET, MILVUS_URI, MILVUS_TOKEN
secrets = [modal.Secret.from_name("neatstandards-secrets")]

# 3. The Web Server
@app.function(
    image=image,
    secrets=secrets,
    scaledown_window=300, # Keep container alive for 5 mins to reduce cold starts
    timeout=600,
)
@modal.web_server(port=8000)
def run():
    # Set the working directory (same as WORKDIR /app)
    os.chdir("/root/app")
    
    # NOTE: We skip 'entrypoint.sh' and run Chainlit directly.
    # If your entrypoint.sh does complex setup, let me know!
    cmd = "chainlit run app.py --host 0.0.0.0 --port 8000 --headless"
    
    subprocess.Popen(cmd, shell=True).wait()

# 4. Database Population Helper
# Since you can't rely on 'compose.yml' to start Milvus, use this to populate
# your remote Zilliz/Milvus instance.
@app.function(image=image, secrets=secrets)
def populate_db():
    os.chdir("/root/app")
    print("Starting database population...")
    subprocess.run(["python", "populate_db.py"], check=True)
    print("Database population complete!")