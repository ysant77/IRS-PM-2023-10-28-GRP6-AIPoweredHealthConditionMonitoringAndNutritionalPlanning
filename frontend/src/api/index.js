import Axios from "axios";

export const hostname = "localhost:8000";

const axiosInstance = Axios.create({
  withCredentials: true,
  baseURL: "http://" + hostname + "/api/",
  headers: {
    "Content-Type": "application/x-www-form-urlencoded",
  },
});

// handle csrf
// axiosInstance.interceptors.request.use((config) => {
//     config.headers['X-Requested-With'] = 'XMLHttpRequest'
//     const regex = /.*csrftoken=([^;.]*).*$/
//     config.headers['X-CSRFToken'] = document.cookie.match(regex) === null ? null : document.cookie.match(regex)[1]
//     return config
// })

axiosInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export const axios = axiosInstance;