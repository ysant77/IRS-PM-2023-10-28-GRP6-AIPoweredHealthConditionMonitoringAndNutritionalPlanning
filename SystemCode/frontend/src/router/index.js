// Composables
import { createRouter, createWebHistory } from "vue-router";

const routes = [
  {
    path: "/",
    component: () => import("@/layouts/default/Default.vue"),
    children: [
      {
        path: "/chat/:cid",
        name: "Chat",
        // route level code-splitting
        // this generates a separate chunk (about.[hash].js) for this route
        // which is lazy-loaded when the route is visited.
        component: () => import("@/views/Chat.vue"),
        props: true,
      },
      {
        path: '',
        name: 'NewChat',
        redirect: '/chat/new',
      },
    ],
  },

  {
    path: "/login/",
    component: () => import("@/layouts/default/Login.vue"),
    name: "login",
  },

  {
    path: "/info/",
    component: () => import("@/layouts/default/Info.vue"),
    name: "info",
  },

  {
    path: "/config-telegram/",
    component: () => import("@/layouts/default/Config.vue"),
    name: "configTelegram",
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
