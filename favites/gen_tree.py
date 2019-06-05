
import subprocess

docker_image_name = 'niemasd/favites:latest'
command = ['docker', 'run', docker_image_name]

subprocess.call(command)