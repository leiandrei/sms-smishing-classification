FROM mcr.microsoft.com/dotnet/sdk:11.0 AS build
WORKDIR /src

COPY SmishingWebApp.csproj ./
RUN dotnet restore SmishingWebApp.csproj

COPY . ./
RUN dotnet publish SmishingWebApp.csproj -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:11.0 AS runtime
WORKDIR /app

COPY --from=build /app/publish ./

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "SmishingWebApp.dll"]