from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/contact-us")
def contact_us():
    return {"email":"abir@example.com","phone":"1234567890"}
