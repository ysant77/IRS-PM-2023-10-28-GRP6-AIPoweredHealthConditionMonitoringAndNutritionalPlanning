## Frontend server setting up and starting (in develop mode)

0. Node.js is required. [Download here.](https://nodejs.org/en/download)

   Open Command Prompt (do not use anaconda shell). Check installation using:

```
npm -v
```

1. Change directory to Systemcode/frontend. Under this directory, run the following command in console to install required node packages.

   A directory called `node_modules` should appear once the installation succeeded. (Size is around 90+ MiB)

```
npm i
```
2. After installed packages, run this to start the server.

```
npm run dev
```
3. visit `localhost:3000` using browser (Chrome or Firefox recommended).

## Note:

Due to the optimized loading method of Vite, web pages may malfunction when loaded for first. Revisiting/reloading the pages should solve the problem.



---

## Credits

Using [**Vue.js 3**](https://github.com/vuejs/) as frontend framework.

Using [**Vue Router**](https://github.com/vuejs/router) for url routing.

Using [**Vuetify 3**](https://github.com/vuetifyjs/vuetify) for UI.

Using [**Vuetify Sonner**](https://github.com/wobsoriano/vuetify-sonner) for UI toast bars.
