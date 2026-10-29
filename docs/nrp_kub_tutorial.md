# NRP Tutorial

1. Log in to NRP at https://nrp.ai/
	- Following the instructions at https://nrp.ai/documentation/userdocs/start/getting-started/
    - Confirm you have access to the "ucsd-ubellur-ugcapstone" namespace under `My account > Namespaces`. 
        - If you do not, contact your TA and Dr. Bellur to add you.
2. Install client tools and verify connection
	- Install kubectl, the kubectl-oidc_login (kubelogin) plugin, and the nrp config file on your computer.
    - Make sure to install the official NRP configuration from the getting started page above.
    - Verify 
		- `kubectl get nodes` provides a very long output
		- `kubectl get pods -n ucsd-ubellur-ugcapstone` Will likely yield "No resources found," this means you have an active connection.
3. Request an interactive JupyterHub server
	- Access the kubernetes JupyterHub Resources at https://nrp.ai/documentation/userdocs/start/using-nautilus/
	- request the compute and memory appropriate for your work that day
	- Please review cluster policies at https://nrp.ai/documentation/userdocs/start/policies/
		+ Of note: Interactive pods (jupyterhub) is limited to 2 GPUs, 32 GB RAM, and 16 CPU cores
4. Non-interactive jobs for more compute 
	1. Create a job manifest file like: "job-example.yaml" below:
		```
		apiVersion: batch/v1
		kind: Job
		metadata:
		  name: training-job
		  namespace: ucsd-ubellur-ugcapstone
		spec:
		  backoffLimit: 1
		  template:
		    spec:
		      containers:
		      - name: worker
		        image: nvcr.io/nvidia/pytorch:24.04-py3
		        command: ["python", "-c", "import torch; print(torch.cuda.device_count())"]
		        resources:
		          requests:
		            cpu: "8"
		            memory: "32Gi"
		            nvidia.com/gpu: "1"
		          limits:
		            cpu: "8"
		            memory: "32Gi"
		            nvidia.com/gpu: "1"
		        env:
		        - name: MY_VAR
		          value: "example"
		      restartPolicy: Never
		```
		- This job runs the command
			```
			python -c import torch; print(torch.cuda.device_count())
			```
		- The activity is defined by image, command / args. Example alternative: 
			```
			command: ["bash", "-c"]
			args: ["python /workspace/train.py"]
			```
	2. Submit the job 
	```
	kubectl apply -f job-example.yaml -n ucsd-ubellur-ugcapstone
	```
	3. Monitor progress
		```
		kubectl get jobs -n ucsd-ubellur-ugcapstone
		kubectl get pods -n ucsd-ubellur-ugcapstone -l job-name=<job-name>
		kubectl logs -f <pod-name> -n ucsd-ubellur-ugcapstone
		```
	3. Collect outputs to your Domain's Persistent Volume if applicable, or use `kubectl cp` to bring files out of the completed pod.
	4. **Clean up**  
	`kubectl delete job <job-name> -n ucsd-ubellur-ugcapstone`
	
5. Shutting down your resources cleanly 
	- (Do not leave an interactive tab open while laptop is asleep)
	1. From the **JupyterHub UI**
		 - Navigate to File > Hub Control Panel and click the "Stop My Server" button. This terminates the pod and releases GPUs/CPU/RAM.
	 2. For **Non-interactive pods**
		 - List the active pods `kubectl get pods -n ucsd-ubellur-ugcapstone`
		 - Identify your pods (careful not to quit anyone else's)
		 - Use `kubectl delete pod <username> -n ucsd-ubellur-ugcapstone` to release the compute resources
		 - Verify the pod is gone by using `kubectl get pods -n ucsd-ubellur-ugcapstone` again