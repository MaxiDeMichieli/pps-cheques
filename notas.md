El procesamiento de cheques es una actividad que tarda, por lo cual no deberia hacerse en un endpoint de una API. El endpoint lo ejecuta y se crea un background process que lo corra y se pueden ir mirando los resultados.

Ir polleando resultados mientras lleguen.