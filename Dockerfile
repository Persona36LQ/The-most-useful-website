FROM nginx:alpine

COPY --from=builder-react /app/client/build /usr/share/nginx/html

COPY --from=builder-django /app /app

COPY nginx_config/nginx.conf /etc/nginx/nginx.conf

EXPOSE 80