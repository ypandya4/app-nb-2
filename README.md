# Scikit model behind flask app on heroku

## tl;dr

You can deploy your own model by

1. Copying the contents of this repo to a new directory
1. Replace `pipeline.pickle`, `dtypes.pickle`, and `columns.json` with
   your own
1. [Deploy to heroku](https://github.com/LDSSA/heroku-model-deploy#deploy-to-heroku)

You'll probably run into a few issues along the way which is why you'll at least want to
skim the contents of the notebooks and this README, in order to have an idea of
where to look when you hit a bump in the road.

## Intro

This is a very simplistic yet effective way to deploy a scikit binary
classifier behind a HTTP server on heroku.

There are 4 main topics to cover here:

1. Serialization
    - This is covered in notebooks
1. Flask
    - Covered here in the README
1. Database connection
    - Covered here in the README
1. Deployment to heroku
    - Also covered here in the README

## Before continuing

Topic #1 is the only one that is not covered here in this README. It is covered in two notebooks
that you must read before moving on with the rest of this README.

[Notebook #1](https://github.com/LDSSA/heroku-example/blob/master/Step%201%20-%20Train%20and%20Serialize.ipynb) has
to do with training and serializing a scikit model as well as how to prepare a new observation
that arrives for prediction.

[Notebook #2](https://github.com/LDSSA/heroku-example/blob/master/Step%202%20-%20Deserialize%20and%20use.ipynb) has
to do with deserialization so that you can re-use a model on new observations without having to re-train
it.

## Python virtual environment

You've probably noticed that we have two requirement files in this repo: `requirements_dev.txt` and `requirements_prod.txt`.

The `requirements_dev.txt` file has the packages that are needed while working on the predictive model, which include jupyter and matplotlib.

The `requirements_prod.txt` file has the packages that are needed when we deploy our model. At that time, we won't need jupyter or matplotlib, so we can save some resources by not installing them.

Now go ahead and create a python virtual env using the requirements in `requirements_dev.txt`, in order to follow this tutorial.

## Flask

Have you already read and understood the notebooks on serialization? Have you already tested your understanding
by pickling and un-pickling your scikit model? Yes yes? Alrighty then, you may continue.

### What is flask

[Flask](http://flask.pocoo.org/) is an HTTP micro-framework. It is a very minimal code library that allows
for quick and simple HTTP server development and is a great alternative to bigger frameworks like Django.
However, be wary before moving forward with a big project using flask - it can get out of hand very quickly
without the enforced structure that other heavier frameworks like Django provide.

For us, since we only need a total of two endpoints (an endpoint is the URL that is used to request an action from the server; in our case we will need two types of actions: requesting a prediction for an observation, and updating an observation's true class) and it doesn't even need to be [RESTful](https://en.wikipedia.org/wiki/Representational_state_transfer), we can stick with
flask and be reasonably justified in it.

### First steps

#### Get a project started

In order to use flask, you will need to be writing some code in a regular
python file - no more notebooks here.

The first step (assuming you have already
created an virtual environment and installed the requirements in `requirements.txt`), is to import flask at the top of the file. 
Let's pretend that we are working in a file called `app.py` in our newly created 
virtual environment.

```py
# the Flask object is for creating an HTTP server - you'll
# see this a few lines down.
# the request object does exactly what the name suggests: holds
# all of the contents of an HTTP request that someone is making
# the jsonify function is useful for when we want to return
# json from the function we are using.
from flask import Flask, request, jsonify

# here we use the Flask constructor to create a new
# application that we can add routes to
app = Flask(__name__)
```

This server doesn't do anything yet. In order to make it do stuff we will
need to add HTTP endpoints to it.

### Making HTTP endpoints

With flask, creating an HTTP endpoint is incredibly simple, assuming that we already
have the `app` object created from the `Flask` constructor. Let's make a single
endpoint that will serve the predictions:

```py
@app.route('/predict', methods=['POST'])
def predict():
    prediction = 0.5
    return jsonify({
        'prediction': prediction
    })
```

The above route that we have isn't very smart in that it returns the same
prediction every time (0.5) and it doesn't actually care about the input
that you sent it. But hey, with just a few lines of code we've almost created an entire server that serves a prediction!

### Making a complete server

Putting it all together with a few lines of code at the end (in order to start
the server in development mode), we've created an entire server that 
can be run by executing `python app.py`:

```py
# these contents can be put into a file called app.py
# and run by executing:
# python app.py

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    prediction = 0.5
    return jsonify({
        'prediction': prediction
    })
    
if __name__ == "__main__":
    app.run(debug=True)

```

If you want to experiment with sending requests to your server, open a new terminal window and execute the following command, in order to get a prediction:

```bash
~ > curl -X POST http://localhost:5000/predict
{
  "prediction": 0.5
}
```

Alright, now that we can run a full flask server, let's try to make something a bit more
useful by receiving new data.

### Receiving a new observation

So now that we've got a way to build an entire server, let's try to actually use the
server to receive new information. There's a pretty nice way to do this via the
[get_json](http://flask.pocoo.org/docs/0.12/api/#flask.Request.get_json) flask function.

For this server, let's say that the model only takes a single field called `unemployed`
and returns `true` if `unemployed` is true and `false` otherwise. The server would now
look like this:

```py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    at_risk = payload['unemployed']
    return jsonify({
        'prediction': at_risk
    })

if __name__ == "__main__":
    app.run(debug=True)
```

You can see the output with the following examples:
```bash
~ > curl -X POST http://localhost:5000/predict -d '{"unemployed": true}' -H "Content-Type:application/json"
{
  "prediction": true
}
```

```bash
~ > curl -X POST http://localhost:5000/predict -d '{"unemployed": false}' -H "Content-Type:application/json"
{
  "prediction": false
}
```

Take a quick note that we had to supply a header of `Content-Type:application/json`
and json data of `{"unemployed": false}`.

### Integrating with a scikit model

Now that we know how to get a python dictionary via the flask `get_json`
function, we're at a point in which we can pick up where the last tutorial
notebook left off! Let's tie it all together by:

1. Deserializing the model, columns, and dtypes
1. Turn the new observation into a pandas dataframe
1. Call `predict_proba` to get the likelihood of survival of the new observation

#### Deserialize model, prep observation, predict

```py
import joblib
import json
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)


with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')


@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    obs = pd.DataFrame([payload], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    return jsonify({
        'prediction': proba
    })


if __name__ == "__main__":
    app.run(debug=True)
```

Check out how we have now taken the payload and turned it into
a new observation that is a single entry in a dataframe,
and can be consumed by the pipeline to be turned into a prediction
of survival. You can see the output with the following:

```
~ >  curl -X POST http://localhost:5000/predict -d '{"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}' -H "Content-Type:application/json"
{
  "prediction": 0.16097398111735517
}
```

## Keeping track of your predictions

Okay, now that you can get data, produce predictions, and return those predictions,
you will need to keep track of what you've been saying about who.
Said another way: you can't just provide predictions and then just forget about it all. You need to
take record of what you have predicted about who, so that later on you can do some additional analysis on your "through the door" population.

In order to do this, we will need to start working with a database. The database
will keep track of the observations, the predictions we have provided for them,
and the true outcomes (should we be luckly enough to find out about them).

### ORMs and peewee

When working with databases in code, you generally want to be using a layer of abstraction
called an [ORM](https://en.wikipedia.org/wiki/Object-relational_mapping). For this
exercise we will use a very simplistic ORM called [peewee](http://docs.peewee-orm.com/en/latest/index.html).
This will allow us to use a local database called [sqlite](https://en.wikipedia.org/wiki/SQLite) (which is basically a file)
when we are developing on our laptops, and use a more production-ready database called
[postgresql](https://en.wikipedia.org/wiki/PostgreSQL) when deploying to heroku, with very
little change to our code.

One cool thing that ORMs allow us to do is define the data model that we want
to use in code. So let's use peewee to create a data model to keep track of
predictions and the probabilities we have assigned to them. Once again, we can
take care of this in a few lines of code:

```py
from peewee import (
    SqliteDatabase, Model, IntegerField,
    FloatField, TextField,
)

DB = SqliteDatabase('predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)
```

Now we need to take a moment to understand exactly how much these
few lines of code have done for us because it is A LOT.

#### Connect to database

`DB = SqliteDatabase('predictions.db')`

Create a sqlite database that will be stored in a file called `predictions.db`.
This may seem trivial right now, but soon enough you will see that changing
out this line of code for one other will result in a lot of value for the effort.

#### Define the data model

`Class Prediction(Model)...`

Define the data model that we will work with. The model has sections for
the following:

- `observation_id`
    - There must be a unique identifier to all observations and it is
      the responsibility of the person providing the observation to give
      this id.
- `observation`
    - We should record the observation itself when it comes, in case
      we want to retrain our model later on.
- `proba`
    - The probability of survival that we assigned to the observation.
- `true_class`
    - This is for later on, in case we find out what actually happened to the observation for which we supplied a prediction.

#### Create the table

`DB.create_tables([Prediction], safe=True)`

The model that we specified must correspond to a database table.
Creation of these tables is something that is it's own non-trivial
headache, and this one line of code makes it so that we don't have to worry about any of it.

## Integrate data model with webserver

Now that we have a webserver and a data model that we are happy with, the next question is how do we put them together? It's actually pretty straightforward!

```py
import joblib
import json
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)


with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


pipeline = joblib.load('pipeline.pickle')


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    _id = obs_dict['id']
    observation = obs_dict['observation']
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True)

```

One piece of the code above that might not be clear at first is:

```py
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data,
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
```

What is this code doing? When we receive a new prediction request, we want to store such request
in our database (to keep track of our model performance). With peewee, we save a new Prediction (basically
a new row in our table) with the `save()` method, which is very neat and convenient.

However, because our table has a unique constraint (no two rows can have the same `observation_id`, which is a unique field),
if we perform the same prediction request twice (with the same id) the system will crash because pewee can't save
again an already saved observation_id; it will throw an `IntegrityError` (as in, we would be asking pewee to violate
the integrity of the table's unique id requirement if we saved a duplicated id, right?).

To avoid that, we do a simple try/except block: if we try a request with the same observation_id, peewee will raise the integrity error and we'll catch it, print a nice error message, and do a database rollback (to close the current save transaction that has failed).

Once your app is setup like this, you can test this with the following command:

```bash
~ > curl -X POST http://localhost:5000/predict -d '{"id": 0, "observation": {"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}}' -H "Content-Type:application/json"
{
  "proba": 0.16097398111735517
}
```

Now let's take note of the few things that changed:

1. The structure of the json input changed. It now includes two top level entries:
    - `id` - This is the unique identifier of the observation;
    - `observation` - This is the actual observation contents that will be sent through
      the pipeline we have un-pickled.
1. We create an instance of `Prediction` with the 3 fields that we care about.
1. We call `save()` on the prediction to save it to the database.
1. We return `proba` so that the caller of the HTTP endpoint knows what you are
saying about the observation.

## Receiving updates

Now that we have a way to provide predictions AND keep track of them, we should
take it to the next level and provide ourselves with a way to receive updates
on observations that we have judged with our predictive model.

We can do this with one extra endpoint that is very straightforward and only
introduces one new concept: database querying through the ORM.

```py
@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})
```

Assuming that we have already processed an observation with id=0, we
can now receive and record the true outcome. Imagine that it is discovered
later on that the person with id=0 didn't survive the titanic disaster. They
would probably enter something into a content management system that
would then trigger a call to your server which would end up looking like
the following:

```bash
~ > curl -X POST http://localhost:5000/update -d '{"id": 0, "true_class": 0}'  -H "Content-Type:application/json"
{
  "id": 1,
  "observation": "{\"id\": 0, \"observation\": {\"Age\": 22.0, \"Cabin\": null, \"Embarked\": \"S\", \"Fare\": 7.25, \"Parch\": 0, \"Pclass\": 3, \"Sex\": \"male\", \"SibSp\": 1}}",
  "observation_id": 0,
  "proba": 0.16097398111735517,
  "true_class": 0
}
```

Similarly to when we saved the prediction requests, we validate that the observation_id we want to update actually exists.

Now to wrap it all up, the way that we can interpret this sequence of events is the following:

1. We provided a prediction of 0.161 probability of survival;
1. We found out later that the person didn't survive.


## Deploy to Heroku

It's cool and all that we can run the servers on our own machines. However, it doesn't
do much good in terms of making the model available to the rest of the world. All this
`localhost` stuff doesn't help anybody that's not typing on your local machine.

So let's take all of the work we've done getting this running and put it on heroku, where it can generate real business value. For this part, you can use any server
that has a static IP address though since we want to avoid the overhead of administering
our own server, we will use a service to do this for us called [heroku](https://www.heroku.com/).
This is one of the oldest managed platforms out there and is quite robust, well-known, and
documented. However, be careful before you move forward with a big project on Heroku -
it can get CRAZY expensive REALLY fast.

However, for our purposes, they offer a free tier webserver and database that is enough to suit our
needs and we can do deployments in a few commands super easily. This may be a bit tough
for some of you but trust me: the alternative of admining your own server is MUCH more difficult.

### Sign up and set up at heroku

Go to the [signup page](https://signup.heroku.com/) and register for the free tier.

Once this is all done, go to the [dashboard](https://dashboard.heroku.com/apps) and create a new
app:

![create new app](https://i.imgur.com/SYyFMV1.png)

Then on the next screen, give it a name and make sure that it's in the Europe zone. It won't
kill anobody to have it in the land of the free but it's kinda far...

![select name and region](https://i.imgur.com/oUPNzOk.png)

Once this is done, select "create app" and you'll be sent to a page that's a bit intimidating
because it just has a lot of stuff. Don't worry though, it's pretty simple what we need
to do next.

First up, make sure that you select the Heroku Git deployment method. It should already be selected
so I don't think you'll need to do anything.

![heroku git](https://i.imgur.com/xt0dAhq.png)

One last bit is missing here: the database. We are going to use a big boy database
called postgresql and luckily heroku has a free tier that allows you to store
up to 10,000 entries which is enough for our purposes (this means that you should try to be conservative
with how you connect to the app and dont go crazy with it, if the database gets full your app will stop working!). You can check heroku's postgresql guide [here](https://devcenter.heroku.com/articles/heroku-postgresql).

To add the database, navigate to `Resources` and search for `postgres`, then select `Heroku Postgres` and the
`Hobby dev - free` tier:

![add postgres](https://i.imgur.com/rZvNnuB.png)


### Let's deploy the titanic model

Let's deploy the server that's contained in this repository. The code is in `app.py` and
there's a few other files that are required but we'll go over those a bit later.

First step toward deployment is to make sure that this repo is cloned on your local
machine.
Since you'll need to change the contents of this repo, and you shouldn't be sharing code with your coleagues, you need to have a private copy of it. For that, go throught the following steps:

- Create a new repository on GitHub. Make it private. You **don't** need to initialize it with any files (as we'll be copying all the files from this repo).
- Create a bare clone of this repo by running the following command:
```bash
~ > git clone --bare git@github.com:LDSSA/heroku-model-deploy.git
```
- Push the contents of this repo to your copy (replace the repo URL with the right one):
 ```bash
~ > cd heroku-model-deploy.git
~ > git push --mirror git@github.com:youruser/new-repo.git
```
- Remove the bare clone of this repository, that you created earlier:
 ```bash
~ > cd ..
~ > rm -rf heroku-model-deploy.git
```
- Clone your copy of this repo:
 ```bash
~ > git clone git@github.com:youruser/new-repo.git
```

Once this is done, you will want to download and install the
[heroku cli](https://devcenter.heroku.com/articles/heroku-cli).

After the heroku cli is installed, you'll need to open a command prompt and
log in. You will use the same credentials that you use to log in through the
web interface with, and it should look something like the following (part of
which is asking you to open up a browser and log in):

```bash
~ > heroku login
heroku: Press any key to open up the browser to login or q to exit:
Opening browser to https://cli-auth.heroku.com/auth/browser/.........
Logging in... done
Logged in as sam@puppiesarecute.com
```

Great! Now when you execute commands on your local machine, the heroku cli will know
who you are!

Next, you will want to navigate on the command line to the repo you've just created (which is the copy of the heroku-mode-deployment repo). It should look something like this:

```bash
~ > cd ldssa/heroku-model-deploy-copy/
~ > ls
Dockerfile                         Step 1 - Train and Serialize.ipynb columns.json                       pipeline.pickle
LICENSE                            Step 2 - Deserialize and use.ipynb dtypes.pickle                      requirements.txt
README.md                          app.py                             heroku.yml                         titanic.csv
```

Make sure that heroku knows about the app you just created by adding a git
remote. Execute the following command but replacing "heroku-model-deploy"
with the name of the app you just created:

```bash
~ > heroku git:remote -a heroku-model-deploy
set git remote heroku to https://git.heroku.com/heroku-model-deploy.git
```

At this point, we'll need to do something a bit extra, in order to sort out our python dependencies in heroku, by specifying and configuring a heroku buildpack (the output might be slightly different, but it will be clear if the command was successful!):

```sh
~ > heroku stack:set container
Stack set. Next release on ⬢ heroku-model-deploy will use container.
Run git push heroku master to create a new release on ⬢ heroku-model-deploy.
```

Now we can push to heroku and our app will be deployed **IN THE CLOUD, WAAAT?!?!**
It is important to remember that only
the changes that you have commited (with git add & git commit) will be deployed. So if you
change your pipeline and retrain a different model you'll need to commit the changes before
pushing to heroku.

```bash
~ > git push heroku master
Counting objects: 3, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 359 bytes | 359.00 KiB/s, done.
Total 3 (delta 2), reused 0 (delta 0)
remote: Compressing source files... done.
remote: Building source:
remote: === Fetching app code
remote:
remote: === Building web (Dockerfile)
remote: Sending build context to Docker daemon  562.2kB
remote: Step 1/5 : FROM python:3.8-buster
remote: 3.8-buster: Pulling from library/python
remote: e4c3d3e4f7b0: Pulling fs layer
remote: 101c41d0463b: Pulling fs layer
remote: 8275efcd805f: Pulling fs layer

...

remote: f327bba4a101: Pushed
remote: 3361b17285b1: Pushed
remote: latest: digest: sha256:8578ff2473fbbafc1bfd14c623102d3a9cb9be0995362f6b054ca4f2f317f39e size: 2639
remote:
remote: Verifying deploy... done.
To https://git.heroku.com/heroku-model-deploy.git
   12b761e..0ad47bf  master -> master
```
 And boom! We're done and deployed! You can actually see this working by executing
 some of the curl commands that we saw before but using `https://<your-app-name>.herokuapp.com`
 rather than `http://localhost` like we saw earlier. For my app it looks like the following:

 ```bash
 ~ > curl -X POST https://heroku-model-deploy.herokuapp.com/predict -d '{"id": 0, "observation": {"Age": 22.0, "Cabin": null, "Embarked": "S", "Fare": 7.25, "Parch": 0, "Pclass": 3, "Sex": "male", "SibSp": 1}}' -H "Content-Type:application/json"
{
  "proba": 0.16097398111735517
}
 ```

 And we can receive updates like the following:

 ```bash
~ > curl -X POST https://heroku-model-deploy.herokuapp.com/update -d '{"id": 0, "true_class": 1}' -H "Content-Type:application/json"
{
  "id": 1,
  "observation": "{\"id\": 0, \"observation\": {\"Age\": 22.0, \"Cabin\": null, \"Embarked\": \"S\", \"Fare\": 7.25, \"Parch\": 0, \"Pclass\": 3, \"Sex\": \"male\", \"SibSp\": 1}}",
  "observation_id": 0,
  "proba": 0.16097398,
  "true_class": 1
}
```

How does all of this work you ask? Well, it has to do with the fact
that there exists a `Dockerfile` and `heroku.yml` files in this repo.
Just be sure that both of these are in your repo and that they are
committed before you `heroku push` and everything should work. If you
do a deploy and it doesn't work, take a look at the files and see
if you might have done something such as change a filename or something
else that would break the boilerplate assumptions that we've made here.

You can see the logs (which is helpful for debugging) with the `heroku logs` command.
Here are the logs for the two calls we just made:

```bash
~ > heroku logs -n 5
2017-12-27T20:14:59.351793+00:00 app[web.1]: [2017-12-27 20:14:59 +0000] [4] [INFO] Using worker: sync
2017-12-27T20:14:59.359149+00:00 app[web.1]: [2017-12-27 20:14:59 +0000] [8] [INFO] Booting worker with pid: 8
2017-12-27T20:14:59.371891+00:00 app[web.1]: [2017-12-27 20:14:59 +0000] [9] [INFO] Booting worker with pid: 9
2017-12-27T20:15:00.678404+00:00 heroku[web.1]: State changed from starting to up
2017-12-27T20:19:25.944435+00:00 heroku[router]: at=info method=POST path="/predict" host=heroku-model-deploy.herokuapp.com request_id=79138602-5b95-497a-9b69-c2528a2bbfc9 fwd="86.166.46.98" dyno=web.1 connect=0ms service=496ms status=200 bytes=187 protocol=https
2017-12-27T20:20:46.033529+00:00 heroku[router]: at=info method=POST path="/update" host=heroku-model-deploy.herokuapp.com request_id=cc92e857-895d-425b-ab00-a92862e1253e fwd="86.166.46.98" dyno=web.1 connect=1ms service=9ms status=200 bytes=417 protocol=https
```


### Import problems

If you are using something like a custom transformer, and getting an import error having to do with your custom code
when unpickling, you'll need to do the following.

#### Put your custom code in a package

Let's say that you have a custom transformer, called `MyCustomTransformer`, that is part of your
pickled pipeline. In that case, you'll want to create a [python package](https://www.learnpython.org/en/Modules_and_Packages)
from which you import the custom transformer, in both your training and deployment code.

In our example, let's create the following package called `custom_transformers` by just creating a directory
with the same name and putting two files inside of it so that it looks like this:

```bash
└── custom_transformers
    ├── __init__.py
    └── transformer.py
```

And inside of `transformer.py` you can put the code for `MyCustomTransformer`. Then, in your training code, you can import them with:

```py
from custom_transformers.transformer import MyCustomTransformer
```

When you un-pickle your model, python should be able to find the custom transformer too.

The dependecies from your custom transformer should be added to the two `requirements.txt` files.


### Last few notes

There were are few additional changes to `app.py` and the rest of the repo that we haven't covered yet, so
let's get that out of the way. You probably won't need to know much about them but if you are having
troubleshooting issues, knowing the following may come in handy.

#### The db connector

When our app is running on heroku, we want to connect to a postgres database, rather than to a sqlite one.
Thus, we had to change the code related to our SqliteDatabase with something that takes care of this:


```py
import os
from playhouse.db_url import connect

# the connect function checks if there is a DATABASE_URL env var
# if it exists, it uses it to connect to a remote postgres db
# otherwise, it connects to a local sqlite db stored in the predictions.db file
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')
```

#### Connecting to the Postgres DB

In case you want to connect to the postgres db in heroku, in order to check what is being stored there, you can find out the connection paremeters with this command, as explained [here](https://devcenter.heroku.com/articles/heroku-postgresql#external-connections-ingress):

```bash
~ > heroku pg:credentials DATABASE
```

### Heroku useful snippets

Push new code after committing it:

```bash
git push heroku master && heroku logs --tail
```

Restart the server:

```bash
heroku ps:restart && heroku logs --tail
```

Check the latest 300 logs of your application:

```bash
heroku logs -n 300
```
