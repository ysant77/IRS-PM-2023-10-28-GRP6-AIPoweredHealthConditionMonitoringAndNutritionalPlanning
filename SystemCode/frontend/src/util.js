import { toast } from "vuetify-sonner";

export function alertToast(msg, color){
    toast(msg, {
    cardProps: {
      color: color,
      duration: 2000,
    },
  });
}