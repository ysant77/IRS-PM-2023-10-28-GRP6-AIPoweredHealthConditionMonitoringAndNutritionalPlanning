import { toast } from "vuetify-sonner";

// export function scrollDown() {
//   var el = document.getElementsByClassName("chat-line");
//   el[el.length - 1].scrollIntoView();
// };

export function alertToast(msg, color){
    toast(msg, {
    cardProps: {
      color: color,
      duration: 2000,
    },
  });
}

export const testdata = [
  {
    sender: "bot",
    text: "Hi! This is prompt message.",
    entries: ["ASDASd", "eqeqfqef"],
    options: ["opt 1", "opt 2", "opt 333"],
    table: {
      header: "diagnosis",
      content: [
        ["a", "b", "c", "d", "e", "f"],
        ["a", "b", "c", "d", "", "f"],
      ],
    },
    plan: [
      {
        meals: [
          { name: "Dinner", items: "asfeaff", energy: "123" },
          { name: "Dinner", items: "asfeaff", energy: "123" },
        ],
        total: [123,456]
      },
      {
        meals: [
          { name: "Dinner", items: "asfeaff", energy: "123" },
          { name: "Dinner", items: "asfeaff", energy: "123" },
        ],
        total: [123,456]
      },
    ],
    suffix: "diagnosis",
    confirm: ["Yes", "No"],
  },

  {
    sender: "bot",
    text: "Hi! This is prompt message.",
    dropdown: ["afeqfqededededededededededed", "qefewfwAEFRAGWRGGGGGGGGG", "wefewf"],
  },
];