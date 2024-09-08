import google.auth

credentials, project = google.auth.default()

print(f"Credenciales: {credentials}")
print(f"Proyecto: {project}")