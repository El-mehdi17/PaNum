const { MongoClient } = require('mongodb');

const uri = "mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/sample_mflix";

const client = new MongoClient(uri);

async function run() {
  try {

    await client.connect();

    const db = client.db("sample_mflix");
    const movies = db.collection("movies");

    const film = await movies.findOne();

    console.log(film);

  } finally {
    await client.close();
  }
}

run();
